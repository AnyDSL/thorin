#include "thorin/continuation.h"
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
        // create a new continuation for every continuation taking a function as parameter
        std::vector<std::pair<Continuation*, Continuation*>> converted;
        for (auto continuation : world_.copy_continuations()) {
            // do not convert empty continuations or intrinsics
            if (!continuation->has_body() || continuation->is_intrinsic()) {
                new_defs_[continuation] = continuation;
                continue;
            }

            auto new_type = world_.fn_type(convert(continuation->type())->ops());
            if (new_type != continuation->type()) {
                //The function type was changed, so the continuation takes another function as a parameter.
                auto new_continuation = world_.continuation(new_type->as<FnType>(), continuation->debug());
                if (continuation->is_intrinsic())
                    new_continuation->set_intrinsic();

                converted.emplace_back(continuation, new_continuation);
                std::cerr << "Creating new conversion: " << continuation->unique_name() << " to " << new_continuation->unique_name() << "\n";
                continuation->type()->dump();
                new_continuation->type()->dump();
                new_defs_[continuation] = new_continuation;
                for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
                    new_defs_[continuation->param(i)] = new_continuation->param(i);
            } else {
                //The type remains unchanged, do not generate a new continuation.
                //We still need to add the continuation to converted, to ensure the jump will be converted later on.
                converted.emplace_back(continuation, continuation);
            }
        }

        // convert the calls to each continuation
        for (auto pair : converted)
            convert_jump(pair.first, pair.second);

        // remove old continuations
        for (auto pair : converted) {
            if (pair.second != pair.first) {
                pair.first->destroy("closure conversion");
                world_.make_internal(pair.first);
            }
        }
    }

    //Convert jump and all arguments.
    void convert_jump(Continuation* source, Continuation* target) {
        assert(source);
        assert(target);

        if (!converted_.emplace(source).second)
            return; //Was already converted once before.

        assert(source->has_body());
        auto body = source->body();

        auto callee = body->callee()->isa_nom<Continuation>();

        if (callee == source) {
            target->jump(callee, body->args(), source->debug());
            return;
        }

        // prevent conversion of calls to vectorize() or cuda(), but allow graph intrinsics
        if (!callee || !callee->is_intrinsic()) {
            Array<const Def*> new_args(body->num_args());
            for (size_t i = 0, e = body->num_args(); i != e; ++i)
                new_args[i] = convert(body->arg(i));
            target->jump(convert(body->callee(), true), new_args, source->debug());
        } else {
            Array<const Def*> new_args(body->num_args());
            for (size_t i = 0, e = body->num_args(); i != e; ++i) {
                if (body->arg(i)->type()->isa<FnType>())
                    new_args[i] = body->arg(i);
                else if (callee->intrinsic() == Intrinsic::Match && i > 2)
                    new_args[i] = body->arg(i);
                else
                    new_args[i] = convert(body->arg(i));
            }
            target->jump(callee, new_args, source->debug());
        }
    }

    const Def* convert(const Def* def, bool as_callee = false) {
        if (auto * source = def->isa_nom<Continuation>()) {
            if (new_defs_.count(def)) def = new_defs_[def];
            auto continuation = def->isa_nom<Continuation>();
            assert(continuation);

            if (as_callee)
                return continuation;
            if (!source->has_body())
                return continuation;
            if (continuation->order() <= 1)
                return continuation;

            convert_jump(source, continuation); //convert_jump must be executed so that continuation has a body.
            assert(continuation->has_body());

            world_.WLOG("slow: closure generated for '{}'", continuation);

            // lift the continuation from its scope
            Scope scope(continuation);
            auto def_set = free_defs(scope, false);
            Array<const Def*> free_vars(def_set.begin(), def_set.end());
            auto filtered_out = std::remove_if(free_vars.begin(), free_vars.end(), [] (const Def* def) {
                assert(!is_mem(def));
                auto continuation = def->isa_nom<Continuation>();
                return continuation && (!continuation->has_body() || continuation->is_intrinsic());
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
            size_t env_param_index = continuation->num_params();
            Array<const Type*> wrapper_param_types(env_param_index + 1);
            for (size_t i = 0, e = continuation->num_params(); i != e; ++i)
                wrapper_param_types[i] = continuation->param(i)->type();
            wrapper_param_types.back() = Closure::environment_type(world_);
            auto wrapper_type = world_.fn_type(wrapper_param_types);
            auto wrapper = world_.continuation(wrapper_type, continuation->debug());

            Array<const Def*> wrapper_args(lifted->num_params());
            const Def* new_mem = wrapper->mem_param();
            if (thin_env) {
                wrapper_args[env_param_index] = world_.cast(free_vars[0]->type(), wrapper->param(env_param_index));
            } else {
                // make the wrapper load the pointer and pass each
                // variable of the environment to the lifted continuation
                auto env_ptr = world_.cast(Closure::environment_ptr_type(world_), wrapper->param(env_param_index));
                auto loaded_env = world_.load(wrapper->mem_param(), world_.bitcast(world_.ptr_type(env_type), env_ptr));
                auto env = world_.extract(loaded_env, 1_u32);
                new_mem = world_.extract(loaded_env, 0_u32);
                for (size_t i = 0, e = free_vars.size(); i != e; ++i)
                    wrapper_args[env_param_index + i] = world_.extract(env, i);
            }
            for (size_t i = 0, e = continuation->num_params(); i != e; ++i) {
                auto param = wrapper->param(i);
                if (param->type()->isa<MemType>()) {
                    // use the mem obtained after the load
                    wrapper_args[i] = new_mem;
                } else {
                    wrapper_args[i] = wrapper->param(i);
                }
            }
            wrapper->jump(lifted, wrapper_args);

            auto closure_type = convert(continuation->type());
            return world_.closure(closure_type->as<ClosureType>(), wrapper, thin_env ? free_vars[0] : world_.tuple(free_vars), continuation->debug());
        } else {
            if (new_defs_.count(def)) return new_defs_[def];
            if (def->isa<Param>() || def->isa<Closure>())
                return def;

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

        // accept one parameter of order 1 (the return continuation) for function types
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
    ContinuationSet converted_;
};


void closure_conversion(World& world) {
    ClosureConversion(world).run();
}

}
