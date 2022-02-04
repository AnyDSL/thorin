#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/mangle.h"

namespace thorin {

class ClosureConversion {
public:
    ClosureConversion(World& world)
        : world_(world)
    {}

    void run() {
        // create a new lambda for every lambda taking a function as parameter
        std::vector<std::pair<Lam*, Lam*>> converted;
        for (auto lam : world_.copy_lams()) {
            // do not convert empty lambdas or intrinsics
            if (!lam->has_body() || lam->is_intrinsic()) {
                new_defs_[lam] = lam;
                continue;
            }

            auto new_type = world_.fn_type(convert(lam->type())->ops());
            if (new_type != lam->type()) {
                auto new_lam = world_.lambda(new_type->as<FnType>(), lam->debug());
                if (lam->is_intrinsic())
                    new_lam->set_intrinsic();

                new_defs_[lam] = new_lam;
                if (lam->has_body()) {
                    auto body = lam->body();
                    for (size_t i = 0, e = lam->num_params(); i != e; ++i)
                        new_defs_[lam->param(i)] = new_lam->param(i);
                    // copy existing call from old lambda
                    new_lam->jump(body->callee(), body->args(), lam->debug());
                    converted.emplace_back(lam, new_lam);
                }
            } else if (lam->has_body()) {
                converted.emplace_back(lam, lam);
            }
        }

        // convert the calls to each lambda
        for (auto pair : converted)
            convert_jump(pair.second);

        // remove old lambdas
        for (auto pair : converted) {
            if (pair.second != pair.first) {
                pair.first->destroy("closure conversion");
                world_.make_internal(pair.first);
            }
        }
    }

    void convert_jump(Lam* lam) {
        assert(lam->has_body());
        auto body = lam->body();
        // prevent conversion of calls to vectorize() or cuda(), but allow graph intrinsics
        auto callee = body->callee()->isa_nom<Lam>();
        if (callee == lam) return;
        if (!callee || !callee->is_intrinsic()) {
            Array<const Def*> new_args(body->num_args());
            for (size_t i = 0, e = body->num_args(); i != e; ++i)
                new_args[i] = convert(body->arg(i));
            lam->jump(convert(body->callee(), true), new_args, lam->debug());
        }
    }

    const Def* convert(const Def* def, bool as_callee = false) {
        if (new_defs_.count(def)) def = new_defs_[def];
        if (def->order() <= 1)
            return def;

        if (auto lam = def->isa_nom<Lam>()) {
            if (!lam->has_body())
                return lam;
            convert_jump(lam);
            if (as_callee)
                return lam;

            world_.WLOG("slow: closure generated for '{}'", lam);

            // lift the lambda from its scope
            Scope scope(lam);
            auto def_set = free_defs(scope, false);
            Array<const Def*> free_vars(def_set.begin(), def_set.end());
            auto filtered_out = std::remove_if(free_vars.begin(), free_vars.end(), [] (const Def* def) {
                assert(!is_mem(def));
                auto lam = def->isa_nom<Lam>();
                return lam && (!lam->has_body() || lam->is_intrinsic());
            });
            free_vars.shrink(filtered_out - free_vars.begin());
            auto lifted = lift(scope, free_vars);

            // get the environment type
            const Type* env_type = nullptr;
            bool thin_env = free_vars.size() == 1 && is_thin(free_vars[0]->type());
            if (thin_env) {
                // optimization: if the environment fits within a pointer or
                // primitive type, pass it by value.
                env_type = free_vars[0]->type();
            } else {
                Array<const Type*> env_ops(free_vars.size());
                for (size_t i = 0, e = free_vars.size(); i != e; ++i)
                    env_ops[i] = free_vars[i]->type();
                env_type = world_.tuple_type(env_ops);
            }

            // create a wrapper that takes a pointer to the environment
            size_t env_param_index = lam->num_params();
            Array<const Type*> wrapper_param_types(env_param_index + 1);
            for (size_t i = 0, e = lam->num_params(); i != e; ++i)
                wrapper_param_types[i] = lam->param(i)->type();
            wrapper_param_types.back() = Closure::environment_type(world_);
            auto wrapper_type = world_.fn_type(wrapper_param_types);
            auto wrapper = world_.lambda(wrapper_type, lam->debug());

            Array<const Def*> wrapper_args(lifted->num_params());
            const Def* new_mem = wrapper->mem_param();
            if (thin_env) {
                wrapper_args[env_param_index] = world_.cast(free_vars[0]->type(), wrapper->param(env_param_index));
            } else {
                // make the wrapper load the pointer and pass each
                // variable of the environment to the lifted lambda
                auto env_ptr = world_.cast(Closure::environment_ptr_type(world_), wrapper->param(env_param_index));
                auto loaded_env = world_.load(wrapper->mem_param(), world_.bitcast(world_.ptr_type(env_type), env_ptr));
                auto env = world_.extract(loaded_env, 1_u32);
                new_mem = world_.extract(loaded_env, 0_u32);
                for (size_t i = 0, e = free_vars.size(); i != e; ++i)
                    wrapper_args[env_param_index + i] = world_.extract(env, i);
            }
            for (size_t i = 0, e = lam->num_params(); i != e; ++i) {
                auto param = wrapper->param(i);
                if (param->type()->isa<MemType>()) {
                    // use the mem obtained after the load
                    wrapper_args[i] = new_mem;
                } else {
                    wrapper_args[i] = wrapper->param(i);
                }
            }
            wrapper->jump(lifted, wrapper_args);

            auto closure_type = convert(lam->type());
            return world_.closure(closure_type->as<ClosureType>(), wrapper, thin_env ? free_vars[0] : world_.tuple(free_vars), lam->debug());
        } else {
            // TODO need to consider Params?
            Array<const Def*> ops(def->ops());
            for (auto& op : ops) op = convert(op);
            return new_defs_[def] = def->rebuild(world_, convert(def->type()), ops);
        }
        THORIN_UNREACHABLE;
    }

    // convert functions to function pointers
    // - fn (A, B, fn(C), fn(D)) => closure(fn (A, B, fn(convert(C)), closure(fn(convert(D)))))
    // - struct S { fn (X, fn(Y)) } => struct T { closure(fn (X, fn(Y))) }
    // - ...
    const Type* convert(const Type* type) {
        if (new_types_.count(type)) return new_types_[type];
        if (type->order() <= 1) return type;
        Array<const Type*> ops(type->ops());

        const Type* new_type = nullptr;
        if (type->isa<StructType>()) {
            // struct types cannot be rebuilt and need to be put in the map first to avoid infinite recursion
            new_type = world_.struct_type(type->as<StructType>()->name(), ops.size());
            new_types_[type] = new_type;
        }

        // accept one parameter of order 1 (the return lambda) for function types
        bool ret = !type->isa<FnType>();
        for (auto& op : ops) {
            op = convert(op);
            if (!ret &&
                op->isa<ClosureType>() &&
                op->as<ClosureType>()->inner_order() == 1) {
                ret = true;
                op = world_.fn_type(op->ops());
            }
        }

        if (type->isa<StructType>()) {
            auto struct_type = new_type->as<StructType>();
            for (size_t i = 0, e = ops.size(); i != e; ++i)
                struct_type->set(i, ops[i]);
        } else {
            new_type = type->rebuild(type->table(), ops);
        }
        if (new_type->order() <= 1)
            return new_types_[type] = new_type;
        else
            return new_types_[type] = world_.closure_type(new_type->ops());
    }

private:
    World& world_;
    Def2Def new_defs_;
    Type2Type new_types_;
};


void closure_conversion(World& world) {
    ClosureConversion(world).run();
}

}
