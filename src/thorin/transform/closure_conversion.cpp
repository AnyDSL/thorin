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
        if (cont->type()->order() % 2 == 1)
            return false;
        return true;
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

    struct UnchangedTypeHelper : Rewriter {
        UnchangedTypeHelper(ClosureConverter& parent) : Rewriter(parent.src(), parent.dst()), parent_(parent) {}

        const Def* rewrite(const Def* odef) override {
            // as long as we're rewriting types, import them as-is
            if (odef->isa<Type>()) {
                return Rewriter::rewrite(odef);
            }
            // if we import something else, its transitive ops will use the parent rewriter
            return parent_.instantiate(odef);
        }

        ClosureConverter& parent_;
    };

    const Type* import_type_as_is(const Type* t) {
        return UnchangedTypeHelper(*this).instantiate(t)->as<Type>();
    }

    struct UnchangedDefHelper : Rewriter {
        UnchangedDefHelper(ClosureConverter& parent, const Def* def) : Rewriter(parent.src(), parent.dst()), def_(def), parent_(parent) {}

        const Def* rewrite(const Def* odef) override {
            if (odef == def_) {
                return Rewriter::rewrite(odef);
            }
            return parent_.instantiate(odef);
        }

        const Def* def_;
        ClosureConverter& parent_;
    };

    const Def* import_def_as_is(const Def* def) {
        return UnchangedDefHelper(*this, def).instantiate(def);
    }

    const Def* rewrite(const Def* odef) override {
        if (auto fn_type = odef->isa<FnType>()) {
            // Turn _all_ FnTypes into ClosureType, we'll undo it where it is specifically OK
            int ret_param = fn_type->ret_param();
            Array<const Type*> rewritten(fn_type->num_ops(), [&](int i) {
                auto old_param_t = fn_type->op(i)->as<Type>();
                if (i == ret_param)
                    return import_type_as_is(old_param_t);
                return instantiate(old_param_t)->as<Type>();
            });

            if (fn_type->order() % 2 == 0)
                return dst().closure_type(rewritten);
            else
                return dst().fn_type(rewritten);

        } /*else if (auto oapp = odef->isa<App>()) {
            auto ncallee = instantiate(oapp->callee());
            auto nfilter = instantiate(oapp->filter())->as<Filter>();
            auto nargs = Array<const Def*>(oapp->num_args(), [&](int i) -> const Def* {
                auto expected_t = ncallee->type()->op(i)->as<Type>();
                auto imported = instantiate(oapp->arg(i));
                if (expected_t != imported->type()) {
                    // the only case we should handle is closures appearing where fns are expected
                    // this rewrite step should not introduce any such nonsense!
                    // TODO: maybe it's simpler to just have the intrinsics (ie vectorize) consume closures...
                    assert(expected_t->isa<FnType>() && imported->type()->isa<ClosureType>());
                    auto closure = imported->as<Closure>();
                    assert(!closure->env()->has_dep(Dep::Param));
                    return closure->fn();
                }
                return imported;
            });
            return dst().app(nfilter, ncallee, nargs, oapp->debug());
        } */else if (auto global = odef->isa<Global>()) {
            auto nglobal = dst().global(import_def_as_is(global->init()), global->is_mutable(), global->debug());;
            nglobal->set_name(global->name());
            if (global->is_external())
                dst().make_external(const_cast<Def*>(nglobal));
            return nglobal;
        } else if (auto ocont = odef->isa_nom<Continuation>()) {
            std::vector<const Type*> nparam_types;

            for (auto pt : ocont->type()->types()) {
                // leave the signature of intrinsics alone!
                if (ocont->is_intrinsic())
                    nparam_types.push_back(import_type_as_is(pt));
                else
                    nparam_types.push_back(instantiate(pt)->as<Type>());
            }

            auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());
            //auto ncont = ocont->stub(*this);

            bool convert = needs_conversion(ocont);
            if (convert) {
                auto closure_type = dst().closure_type(nparam_types);

                for (size_t i = 0; i < ocont->num_params(); i++)
                    insert(ocont->param(i), ncont->param(i));

                Scope scope(ocont);
                std::vector<const Def*> free_vars;
                for (auto free : spillable_free_defs(scope))
                    free_vars.push_back(free);

                if (!free_vars.empty()) {
                    dst().WLOG("slow: closure generated for '{}'", ocont);
                    auto [env_type, thin] = get_env_type(free_vars);
                    env_type = instantiate(env_type)->as<Type>();

                    for (auto fv : free_vars) {
                        dst().VLOG("fv: {} : {}", fv, fv->type());
                    }

                    // create a wrapper that takes a pointer to the environment
                    size_t env_param_index = ocont->num_params();
                    nparam_types.push_back(Closure::environment_type(dst()));
                    auto wrapper_type = dst().fn_type(nparam_types);
                    ncont = dst().continuation(wrapper_type, ocont->debug());

                    Array<const Def*> wrapper_args(ocont->num_params() + free_vars.size());
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

                    auto closure = dst().closure(closure_type, ncont, instantiate(thin ? free_vars[0] : src().tuple(free_vars)), ocont->debug());
                    insert(ocont, closure);
                    auto lifted = lift(scope, free_vars);

                    Scope lifted_scope(lifted);
                    ncont->jump(instantiate(lifted), wrapper_args);
                    assert(lifted_scope.parent_scope() == nullptr);

                    return closure;
                } else {
                    auto closure = dst().closure(closure_type, ncont, dst().tuple({}), ocont->debug());
                    insert(ocont, closure);
                    ncont->rebuild_from(*this, ocont);
                    return closure;
                }
            } else {
                insert(ocont, ncont);
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
