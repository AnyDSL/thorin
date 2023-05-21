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
        if (auto found = should_convert.lookup(cont))
            return found.value();

        if (cont->is_intrinsic() || cont->is_exported())
            return false;

        bool needs_conversion = false;
        src().DLOG("checking for uses of {} ...", cont);
        for (auto use : cont->copy_uses()) {
            if (is_use_first_class(use)) {
                needs_conversion = true;
                break;
            }
        }

        should_convert.insert(std::make_pair(cont, needs_conversion));

        return needs_conversion;
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

    std::vector<const Type*> rewrite_params(Continuation* ocont) {
        std::vector<const Type*> nparam_types;

        for (auto pt : ocont->type()->types()) {
            // leave the signature of intrinsics alone!
            if (ocont->is_intrinsic())
                nparam_types.push_back(import_type_as_is(pt));
            else
                nparam_types.push_back(instantiate(pt)->as<Type>());
        }

        return nparam_types;
    }

    const Continuation* import_continuation_as_is(Continuation* ocont) {
        auto nparam_types = rewrite_params(ocont);
        auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());

        for (size_t i = 0; i < ocont->num_params(); i++)
            insert(ocont->param(i), ncont->param(i));

        insert(ocont, ncont);
        dst().VLOG("no closure generated for '{}' -> '{}'", ocont, ncont);
        ncont->rebuild_from(*this, ocont);
        return ncont;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto fn_type = odef->isa<FnType>()) {
            Array<const Type*> ntypes(fn_type->num_ops(), [&](int i) {
                auto old_param_t = fn_type->op(i)->as<Type>();
                return instantiate(old_param_t)->as<Type>();
            });
            if (fn_type->isa<ReturnType>())
                return dst().return_type(ntypes);

            // Turn all functions into closures, we'll undo it where it is specifically OK
            auto ntype = dst().closure_type(ntypes);
            return ntype;

        } else if (auto global = odef->isa<Global>()) {
            auto nglobal = dst().global(import_def_as_is(global->init()), global->is_mutable(), global->debug());
            nglobal->set_name(global->name());
            if (global->is_external())
                dst().make_external(const_cast<Def*>(nglobal));
            return nglobal;
        } else if (auto ocont = odef->isa_nom<Continuation>()) {
            if (!needs_conversion(ocont))
                return import_continuation_as_is(ocont);

            auto nparam_types = rewrite_params(ocont);
            auto closure_type = dst().closure_type(nparam_types);

            Scope scope(ocont);
            std::vector<const Def*> free_vars;
            for (auto free : spillable_free_defs(scope))
                free_vars.push_back(free);

            if (!free_vars.empty()) {
                dst().WLOG("slow: closure generated for '{}'", ocont);
                auto [env_type, thin] = get_env_type(free_vars);
                env_type = instantiate(env_type)->as<Type>();

                // create a wrapper that takes a pointer to the environment
                size_t env_param_index = ocont->num_params();
                nparam_types.push_back(Closure::environment_type(dst()));
                auto wrapper_type = dst().fn_type(nparam_types);
                auto ncont = dst().continuation(wrapper_type, ocont->debug());

                for (size_t i = 0; i < ocont->num_params(); i++)
                    insert(ocont->param(i), ncont->param(i));

                dst().WLOG("slow: rewriting '{}' as '{}'", ocont, ncont);

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

                auto lifted = lift(scope, free_vars);
                Scope lifted_scope(lifted);
                assert(lifted_scope.parent_scope() == nullptr);

                auto closure = dst().closure(closure_type, ncont, instantiate(thin ? free_vars[0] : src().tuple(free_vars)), ocont->debug());
                insert(ocont, closure);

                ncont->jump(instantiate(lifted), wrapper_args);

                return closure;
            } else {
                auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());

                for (size_t i = 0; i < ocont->num_params(); i++)
                    insert(ocont->param(i), ncont->param(i));

                auto closure = dst().closure(closure_type, ncont, dst().tuple({}), ocont->debug());
                insert(ocont, closure);
                ncont->rebuild_from(*this, ocont);
                return closure;
            }
        } else if (auto app = odef->isa<App>()) {
            auto cont = app->callee()->isa_nom<Continuation>();
            if (cont && !needs_conversion(cont)) {
                Array<const Def*> nargs(app->num_args(), [&](int i) -> const Def* {
                    auto oarg = app->arg(i);
                    auto narg = instantiate(oarg);
                    return narg;
                });
                return dst().app(instantiate(app->callee()), nargs);
            }
        }
        return Rewriter::rewrite(odef);
    }

    bool is_use_first_class(Use& use) {
        if (use.def()->isa<Param>())
            return false;
        if (use.def()->isa<ReturnPoint>())
            return false;
        if (auto app = use.def()->isa<App>()) {
            if (use.index() == App::CALLEE_POSITION)
                return false;
            if (auto callee = app->callee()->isa_nom<Continuation>()) {
                if (callee->is_intrinsic())
                    return false;
            }
        }

        src().DLOG("{} is used as a first-class value in {} ({}, index={})", use.def()->op(use.index()), use.def(), tag2str(use.def()->tag()), use.index());
        return true;
    }

    ContinuationMap<bool> should_convert;
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
