#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/rewrite.h"
#include "closure_conversion.h"

namespace thorin {

struct ClosureConverter {
    ClosureConverter(World& src, World& dst, LiftMode mode) : src_(src), dst_(dst), root_rewriter_(*this, nullptr, nullptr), forest_(src), mode_(mode) {
        assert(&src != &dst);
    }

    DefSet spillable_free_defs(Continuation* entry, DefSet& additional_rebuild) {
        DefSet result;
        unique_queue<DefSet> queue;

        for (auto def: forest_.get_scope(entry).free_frontier())
            queue.push(def);
        entry->world().VLOG("Computing free variables for {}", entry);

        while (!queue.empty()) {
            auto free = queue.pop();
            assert(!free->type()->isa<MemType>());

            //if (free == entry)
            //    continue;

            if (auto ret_cont = free->isa<ReturnPoint>()) {
                additional_rebuild.insert(ret_cont);
                queue.push(ret_cont->continuation());
                continue;
            }

            if (auto cont = free->isa_nom<Continuation>()) {
                if (!should_process(cont) || mode_ == LiftMode::Lift2Cff) {
                    auto& scope = forest_.get_scope(cont);
                    if (scope.parent_scope() != nullptr) {
                        entry->world().VLOG("encountered a basic block from a parent scope: {}", cont);
                        // enforce that this continuation gets rebuilt even though it originally did not belong in the scope
                        additional_rebuild.insert(cont);
                        for (auto oparam : cont->params())
                            additional_rebuild.insert(oparam);
                        // provide what that continuation will need
                        auto& frontier = scope.free_frontier();
                        for (auto def: frontier) {
                            queue.push(def);
                        }
                        continue;
                    }
                    entry->world().VLOG("encountered a top level function: {}", cont);
                    continue;
                }
            }

            if (free->has_dep(Dep::Param)) {
                entry->world().VLOG("fv of {}: {} : {}", entry, free, free->type());
                result.insert(free);
            } else
                free->world().WLOG("ignoring {} because it has no Param dependency", free);
        }

        return result;
    }

    struct ScopeRewriter : public Rewriter {
        ScopeRewriter(ClosureConverter& converter, Scope* scope, ScopeRewriter* parent) : Rewriter(converter.src(), converter.dst()), converter_(converter), parent_(parent), scope_(scope), name_(scope ? scope->entry()->unique_name() : "root") {
            if (parent)
                depth_ = parent->depth_ + 1;
            //assert(depth_ < 4);
        }

        int depth_ = 0;
        ClosureConverter& converter_;
        ScopeRewriter* parent_;
        Scope* scope_;
        DefSet additional_;
        std::vector<std::unique_ptr<ScopeRewriter>> children_;
        std::string name_;

        const Def* lookup(const thorin::Def* odef) override {
            auto found = Rewriter::lookup(odef);
            if (found) return found;
            if (parent_) return parent_->lookup(odef);
            return nullptr;
        }

        const std::string dump() {
            std::string n = name_;
            ScopeRewriter* r = parent_;
            while (r) {
                n = r->name_ + "::" + n;
                r = r->parent_;
            }
            return n;
        }

        const Def* rewrite(const Def* odef) override;

        ScopeRewriter(ScopeRewriter&) = delete;
        ScopeRewriter(ScopeRewriter&&) = delete;
    };

    ScopeRewriter& root_rewriter() {
        return root_rewriter_;
    }

    bool should_process(Continuation* cont) {
        if (auto found = should_convert_.lookup(cont))
            return found.value();

        if (cont->is_intrinsic() || cont->is_exported())
            return false;

        src().DLOG("checking for uses of {} ...", cont);
        bool needs_lifting = false;
        bool needs_conversion = false;
        if (cont->is_returning()) {
            src().DLOG("{} is as a returning continuation !", cont);
            needs_lifting = true;
        }

        for (auto use : cont->copy_uses()) {
            if (is_use_first_class(use)) {
                needs_conversion = true;
            }
            if (needs_conversion)
                break;
        }

        bool should_process;
        switch (mode_) {
            case LiftMode::Lift2Cff:
                should_process = needs_lifting && !needs_conversion;
                break;
            case LiftMode::ClosureConversion:
                should_process = needs_conversion;
                break;
            case LiftMode::JoinTargets:
                should_process = needs_conversion;
                break;
        }

        should_convert_.insert(std::make_pair(cont, should_process));
        return should_process;
    }

    std::tuple<const Type*, bool> get_env_type(ArrayRef<const Def*> free_vars) {
        // get the environment type
        const Type* env_type = nullptr;
        bool thin_env = free_vars.size() == 1 && ClosureType::is_thin(free_vars[0]->type());
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

    std::vector<const Type*> rewrite_params(ScopeRewriter& rewriter, Continuation* ocont) {
        std::vector<const Type*> nparam_types;

        for (auto pt : ocont->type()->types()) {
            const Type* npt = rewriter.instantiate(pt)->as<Type>();
            // in intrinsics, don't closure-convert immediate parameters
            if (ocont->is_intrinsic() && pt->tag() == Node_FnType) {
                npt = dst().fn_type(npt->as<FnType>()->types());
            }
            nparam_types.push_back(npt);
        }

        return nparam_types;
    }

    Continuation* as_continuation(const Def* ndef) {
        if (auto ncont = ndef->isa_nom<Continuation>())
            return ncont;
        auto wrapper = dst().continuation(dst().fn_type(ndef->type()->as<FnType>()->types()), {ndef->debug().name + "_wrapped" });
        wrapper->jump(ndef, wrapper->params_as_defs());
        return wrapper;
    }

    bool is_use_first_class(Use& use) {
        if (use.def()->isa<Param>())
            return false;
        if (use.def()->isa<Closure>())
            return false;
        if (use.def()->isa<ReturnPoint>())
            return false;
        if (auto app = use.def()->isa<App>()) {
            if (use.index() == App::CALLEE_POSITION)
                return false;
            if (auto callee = app->callee()->isa_nom<Continuation>()) {
                if (callee->is_intrinsic()) {
                    if (mode_ == LiftMode::JoinTargets && callee->intrinsic() == Intrinsic::Control && use.index() == App::ARGS_START_POSITION + 2) {
                        src().DLOG("{} is used as a join target in {}", use.def()->op(use.index()), app);
                        return true;
                    }
                    return false;
                }
            }
        }

        assert(mode_ != LiftMode::JoinTargets && "CC should have been performed earlier...");
        src().DLOG("{} is used as a first-class value in {} ({}, index={})", use.def()->op(use.index()), use.def(), tag2str(use.def()->tag()), use.index());
        return true;
    }

    World& src_;
    World& dst_;
    World& src() { return src_; }
    World& dst() { return dst_; }
    LiftMode mode_;

    ScopeRewriter root_rewriter_;
    ScopesForest forest_;
    ContinuationMap<bool> should_convert_;
    ContinuationMap<std::vector<const Def*>> lifted_env_;
    std::vector<std::function<void()>> todo_;
};

const Def* ClosureConverter::ScopeRewriter::rewrite(const Def* odef) {
    assert(&odef->world() == &src());
    if (parent_) {
        if (!scope_->contains(odef) && !additional_.contains(odef)) {
            //dst().DLOG("Deferring rewriting of {} to {}", odef, parent_->name_);
            return parent_->rewrite(odef);
        }
    }

    if (auto fn_type = odef->isa<FnType>()) {
        if (converter_.mode_ == LiftMode::ClosureConversion || converter_.mode_ == LiftMode::JoinTargets) {
            Array<const Type*> ntypes(fn_type->num_ops(), [&](int i) {
                auto old_param_t = fn_type->op(i)->as<Type>();
                return instantiate(old_param_t)->as<Type>();
            });
            if (fn_type->isa<ReturnType>())
                return dst().return_type(ntypes);
            if (fn_type->isa<JoinPointType>())
                return dst().join_point_type(ntypes);

            // Turn all functions into closures, we'll undo it where it is specifically OK
            auto ntype = dst().closure_type(ntypes);
            return ntype;
        }
    } else if (auto global = odef->isa<Global>()) {
        auto nglobal = dst().global(instantiate(global->init()), global->is_mutable(), global->debug());
        nglobal->set_name(global->name());
        if (global->is_external())
            dst().make_external(const_cast<Def*>(nglobal));
        return nglobal;
    } else if (auto ocont = odef->isa_nom<Continuation>()) {
        dst().DLOG("analysing '{}' in {}", ocont, dump());
        auto& scope = converter_.forest_.get_scope(ocont);
        ScopeRewriter* body_rewriter;
        const Def* ndef = nullptr;
        auto nparam_types = converter_.rewrite_params(*this, ocont);
        if (!converter_.should_process(ocont)) {
            if (converter_.mode_ == LiftMode::JoinTargets && ocont->intrinsic() == Intrinsic::Control) {
                nparam_types[2] = dst().closure_type(nparam_types[2]->as<FnType>()->types());
            }

            auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());
            insert(ocont, ncont);
            ndef = ncont;

            children_.emplace_back(std::make_unique<ScopeRewriter>(converter_, &scope, this));
            body_rewriter = children_.back().get();

            for (size_t i = 0; i < ocont->num_params(); i++)
                body_rewriter->insert(ocont->param(i), ncont->param(i));
            dst().DLOG("no closure generated for '{}' in {}", ocont, dump());
            converter_.todo_.emplace_back([=]() {
                ncont->rebuild_from(*body_rewriter, ocont);
                dst().DLOG("'{}' rebuilt as '{}' in {}", ocont, ncont, dump());
            });
        } else {
            auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());

            // only the root rewriter is a parent, we're remaking everything else
            children_.emplace_back(std::make_unique<ScopeRewriter>(converter_, &scope, &converter_.root_rewriter_));
            body_rewriter = children_.back().get();

            // Compute all the free variables and record additional nodes to be rebuilt in this context
            std::vector<const Def*> free_vars;
            for (auto free : converter_.spillable_free_defs(ocont, body_rewriter->additional_)) {
                if (auto free_cont = free->isa_nom<Continuation>()) {
                    assert(converter_.should_process(free_cont));
                }
                free_vars.push_back(free);
            }

            Closure* closure = nullptr;
            const Param* closure_param = nullptr;
            if (converter_.mode_ == LiftMode::ClosureConversion || converter_.mode_ == LiftMode::JoinTargets) {
                // create a wrapper that takes a pointer to the environment
                auto closure_type = dst().closure_type(nparam_types);
                nparam_types.push_back(closure_type);
                closure_param = ncont->append_param(closure_type, { "self" });

                closure = dst().closure(closure_type, ocont->debug());
                closure->set_fn(ncont);
                ndef = closure;
                insert(ocont, closure);
                body_rewriter->insert(ocont, ncont->params().back());
                dst().WLOG("converting '{}' into '{}' in {}", ocont, closure, dump());
            } else if (!free_vars.empty()) {
                Continuation* lifted = dst().continuation(ncont->type(), ncont->debug());
                for (auto free_var : free_vars) {
                    auto captured = lifted->append_param(instantiate(free_var->type())->as<Type>(), free_var->name() + "_captured");
                    body_rewriter->insert(free_var, captured);
                }
                lifted->jump(ncont, lifted->params_as_defs().get_front(ncont->num_params()));

                body_rewriter->insert(ocont, ncont);
                insert(ocont, lifted);
                ndef = lifted;
                converter_.lifted_env_.emplace(lifted, free_vars);
            } else {
                return Rewriter::rewrite(odef);
            }

            if (!free_vars.empty()) {
                dst().WLOG("slow: rewriting '{}' as '{}' in {}", ocont, ncont, dump());
                auto [env_type, thin] = converter_.get_env_type(free_vars);
                env_type = this->instantiate(env_type)->as<Type>();

                converter_.todo_.emplace_back([=]() {
                    const Def* new_mem = ncont->mem_param();

                    if (converter_.mode_ == LiftMode::ClosureConversion || converter_.mode_ == LiftMode::JoinTargets) {
                        Array<const Def*> instantiated_free_vars = Array<const Def*>(free_vars.size(), [&](const int i) -> const Def* {
                            auto env = instantiate(free_vars[i]);
                            // it cannot be a basic block, and it cannot be top-level either
                            // if it used to be a continuation it should be a closure now.
                            assert(!env->isa_nom<Continuation>());
                            return env;
                        });
                        closure->set_env(thin ? instantiated_free_vars[0] : dst().heap(dst().tuple(instantiated_free_vars)));

                        assert(closure_param);
                        auto env = dst().extract(closure_param, 1);
                        if (thin) {
                            auto captured = dst().cast(this->instantiate(free_vars[0]->type())->as<Type>(), env);
                            captured->set_name(free_vars[0]->name() + "_captured");
                            body_rewriter->insert(free_vars[0], captured);
                        } else {
                            // make the wrapper load the pointer and pass each
                            // variable of the environment to the lifted continuation
                            auto env_ptr = dst().cast(Closure::environment_ptr_type(dst()), env);
                            auto loaded_env = dst().load(ncont->mem_param(), dst().bitcast(dst().ptr_type(env_type), env_ptr));
                            auto env_data = dst().extract(loaded_env, 1_u32);
                            new_mem = dst().extract(loaded_env, 0_u32);
                            for (size_t i = 0, e = free_vars.size(); i != e; ++i) {
                                auto captured = dst().extract(env_data, i);
                                captured->set_name(free_vars[i]->name() + "_captured");
                                body_rewriter->insert(free_vars[i], captured);
                            }
                        }
                    }

                    // Map old arguments to new ones
                    for (size_t i = 0, e = ocont->num_params(); i != e; ++i) {
                        auto param = ncont->param(i);
                        if (ocont->mem_param() == ocont->param(i)) {
                            // use the mem obtained after the load
                            body_rewriter->insert(ocont->mem_param(), new_mem);
                        } else {
                            body_rewriter->insert(ocont->param(i), param);
                        }
                    }

                    ncont->set_body(body_rewriter->instantiate(ocont->body())->as<App>());

                    dst().VLOG("finished body of {}", ncont);
                });
            } else if (converter_.mode_ == LiftMode::ClosureConversion || converter_.mode_ == LiftMode::JoinTargets) {
                dst().DLOG("dummy closure generated for '{}' -> '{}' in {}", ocont, ncont, dump());
                for (size_t i = 0; i < ocont->num_params(); i++)
                    body_rewriter->insert(ocont->param(i), ncont->param(i));

                closure->set_env(dst().tuple({}));
                converter_.todo_.emplace_back([=]() {
                    ncont->rebuild_from(*body_rewriter, ocont);
                });
            }
        }
        assert(ndef);
        return ndef;
    } else if (auto app = odef->isa<App>()) {
        auto ncallee = instantiate(app->callee());
        if (auto ncont = ncallee->isa_nom<Continuation>(); ncont) {
            std::vector<const Def*> nargs;
            nargs.resize(app->num_args());

            for (size_t i = 0; i < app->num_args(); i++) {
                auto oarg = app->arg(i);
                if (ncont->is_intrinsic() && ncont->type()->types()[i]->tag() == Node_FnType)
                    nargs[i] = converter_.as_continuation(instantiate(oarg));
                else nargs[i] = instantiate(oarg);
            };

            if (auto extra_params = converter_.lifted_env_.lookup(ncont); extra_params.has_value()) {
                for (auto extra : extra_params.value()) {
                    nargs.push_back(instantiate(extra));
                }
            }

            return dst().app(ncallee, nargs);
        }
    }
    return Rewriter::rewrite(odef);
}

void validate_all_returning_functions_top_level(World& world) {
    ScopesForest forest(world);
    for (auto cont : world.copy_continuations()) {
        if (cont->type()->is_returning()) {
            auto parent = forest.get_scope(cont).parent_scope();
            if (parent != nullptr) {
                world.ELOG("returning continuation {} is not top-level after closure conversion and belongs to {}'s scope", cont, parent);
                assert(false);
            }
        }
    }
}

void closure_conversion(Thorin& thorin, LiftMode mode) {
    auto& src = thorin.world_container();
    auto dst = std::make_unique<World>(*src);
    ClosureConverter converter(*src, *dst, mode);

    for (auto& external : src->externals())
        converter.root_rewriter().instantiate(external.second);

    while (!converter.todo_.empty()) {
        auto f = converter.todo_.back();
        converter.todo_.pop_back();
        f();
    }

    //validate_all_returning_functions_top_level(*dst);

    src.swap(dst);
}

}
