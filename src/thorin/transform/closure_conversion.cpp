#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

struct ClosureConverter : public Rewriter {
    ClosureConverter(World& src, World& dst) : Rewriter(src, dst) {}

    bool needs_conversion(Continuation* cont) {
        if (cont->is_exported() || !cont->has_body())
            return false;
        if (cont->is_intrinsic())
            return false;

        // basic blocks _never_ need conversion!
        if (cont->type()->order() == 1)
            return false;
        else if (cont->type()->order() == 2) {
            Scope scope(cont);
            // only allow conversion if we're not top level already
            Array<const Def*> free_vars = spillable_free_defs(scope);
            if (free_vars.empty())
                return false;
        }

        for (auto use : cont->uses()) {
            if (use.def()->isa<Param>())
                continue;
            else if (use->isa<App>() && use.index() == App::CALLEE_POSITION)
                continue;
            return true;
        }
        return false;
    }

    std::tuple<const Type*, bool> get_env_type(ArrayRef<const Def*> free_vars) {
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
            env_type = src().tuple_type(env_ops);
        }
        return std::make_tuple(env_type, thin_env);
    }

    const Type* unclosurify_type(const Type* src) {
        if (auto closure_t = src->isa<ClosureType>())
            return dst().fn_type(closure_t->types());
        return src;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto fn_type = odef->isa<FnType>()) {
            // Turn _all_ FnTypes into ClosureType, we'll undo it where it is specifically OK
            Array<const Type*> rewritten(fn_type->num_ops(), [&](size_t i) { return instantiate(fn_type->op(i))->as<Type>(); });
            int ret_param = fn_type->ret_param();
            if (ret_param >= 0)
                rewritten[ret_param] = unclosurify_type(rewritten[ret_param]);
            if (fn_type->order() > 1)
                return dst().closure_type(rewritten);
            else
                return dst().fn_type(rewritten);
        } else if (auto ocont = odef->isa_nom<Continuation>()) {
            std::vector<const Type*> nparam_types;
            for (auto pt : instantiate(ocont->type())->as<FnType>()->types())
                nparam_types.push_back(pt);

            // auto ret_type_i = ocont->type()->ret_param();
            // if (ret_type_i >= 0) {
            //     // leave the ret param alone
            //     nparam_types[ret_type_i] = unclosurify_type(nparam_types[ret_type_i]);
            // }

            if (ocont->is_intrinsic()) {
                for (size_t i = 0; i < ocont->num_params(); i++) {
                    if (nparam_types[i]->isa<ClosureType>() && !ocont->param(i)->type()->isa<ClosureType>()) {
                        nparam_types[i] = unclosurify_type(nparam_types[i]);
                    }
                }
            }

            bool convert = needs_conversion(ocont);

            auto ncont = ocont->stub(*this);
            insert(ocont, ncont);

            if (convert) {
                auto closure_type = dst().closure_type(nparam_types);
                dst().WLOG("slow: closure generated for '{}'", ocont);

                Scope scope(ocont);
                Array<const Def*> free_vars = spillable_free_defs(scope);
                auto [env_type, thin] = get_env_type(free_vars);
                env_type = instantiate(env_type)->as<Type>();
                auto lifted = lift(scope, free_vars);

                // create a wrapper that takes a pointer to the environment
                size_t env_param_index = ocont->num_params();
                nparam_types.push_back(Closure::environment_type(dst()));
                auto wrapper_type = dst().fn_type(nparam_types);
                ncont = dst().continuation(wrapper_type, ocont->debug());

                Array<const Def*> wrapper_args(lifted->num_params());
                const Def* new_mem = ncont->mem_param();
                if (thin) {
                    wrapper_args[env_param_index] = dst().cast(instantiate(free_vars[0]->type())->as<Type>(), ncont->param(env_param_index));
                } else {
                    // make the wrapper load the pointer and pass each
                    // variable of the environment to the lifted continuation
                    auto env_ptr = dst().cast(Closure::environment_ptr_type(dst()), ncont->param(env_param_index));
                    auto loaded_env = dst().load(ncont->mem_param(), dst().bitcast(dst().ptr_type(env_type), env_ptr));
                    auto env_data = dst().extract(loaded_env, 1_u32);
                    new_mem = dst().extract(loaded_env, 0_u32);
                    for (size_t i = 0, e = free_vars.size(); i != e; ++i)
                        wrapper_args[env_param_index + i] = dst().extract(env_data, i);
                }

                // make the wrapper call into the lifted continuation with the right arguments
                for (size_t i = 0, e = ocont->num_params(); i != e; ++i) {
                    auto param = ncont->param(i);
                    if (param->type()->isa<MemType>()) {
                        // use the mem obtained after the load
                        wrapper_args[i] = new_mem;
                    } else {
                        wrapper_args[i] = ncont->param(i);
                    }
                }
                ncont->jump(instantiate(lifted), wrapper_args);

                return dst().closure(closure_type, ncont, instantiate(thin ? free_vars[0] : src().tuple(free_vars)), ocont->debug());
            } else {
                ncont->rebuild_from(*this, ocont);
                return ncont;
            }
        }
        return Rewriter::rewrite(odef);
    }
};

void closure_conversion(Thorin& thorin) {
    auto& src = thorin.world_container();
    auto dst = std::make_unique<World>(*src);
    ClosureConverter converter(*src, *dst);
    for (auto& ext : src->externals())
        converter.instantiate(ext.second);
    src.swap(dst);
}

}
