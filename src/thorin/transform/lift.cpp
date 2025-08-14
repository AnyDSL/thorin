#include "lift.h"


#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

struct ClosureConverter {
    ClosureConverter(World& src, World& dst) : src_(src), dst_(dst), root_rewriter_(*this, nullptr, nullptr), forest_(src) {
        assert(&src != &dst);
    }

    struct ScopeAnalysis {
        ClosureConverter& converter;
        Scope& scope;
        DefSet free_vars;
        DefSet rematerialize;
        bool convert_to_closure = false;

        ScopeAnalysis(ClosureConverter& conv, Scope& s) : converter(conv), scope(s) {}

        void promote_to_closure() {
            // if we have to be closure converted - check out the free variables
            if (!convert_to_closure) {
                convert_to_closure = true;
                converter.spillable_free_defs(scope.entry(), free_vars, rematerialize);
                for (auto fv : free_vars) {
                    if (auto spill_this = fv->isa_nom<Continuation>()) {
                        converter.src().DLOG("{} appears as a free variable in {} and therefore must also be converted", fv, scope.entry());
                        converter.lookup(spill_this).promote_to_closure();
                    }
                }
            }
        }
    };

    ScopeAnalysis& lookup(Continuation* c) {
        auto& found = analysis_[c];
        assert(found);
        return *found.get();
    }

    bool is_acceptable_use_for_bb(Use& use) const {
        if (use->isa<Param>() || use->isa<ReturnPoint>())
            return true;
        if (auto app = use->isa<App>()) {
            // being the callee is OK
            if (use.index() == App::Ops::Callee)
                return true;
            // branch target are OK
            if (auto cont = app->callee()->isa_nom<Continuation>()) {
                if (cont->is_intrinsic())
                    return true;
            }
        }
        //if (auto tuple = use->isa<Tuple>()) {
        //    // if the tuple is consumed appropriately, this is OK
        //    // this is necessary because match() takes tuples nowadays
        //    for (auto tu : tuple->uses()) {
        //        if (!is_acceptable_use_for_bb(tu))
        //            return false;
        //    }
        //    return true;
        //}

        return false;
    }

    void scan() {
        ContinuationSet top;
        forest_.for_each<false>([&](auto& scope) {
            top.insert(scope.entry());
        });
        scan_siblings(top);
    }

    void scan_siblings(const ContinuationSet& siblings) {
        // prepare the analysis for all siblings first
        for (auto entry : siblings) {
            auto& found = analysis_[entry];
            assert(!found);
            Scope& scope = forest_.get_scope(entry);
            found = std::make_unique<ScopeAnalysis>(*this, scope);
        }

        for (auto entry : siblings) {
            Scope& scope = forest_.get_scope(entry);
            scan_scope(scope);
        }
    }

    void scan_scope(Scope& scope) {
        //src().DLOG("Scanning scope: {}", scope.entry());
        ScopeAnalysis& a = lookup(scope.entry());

        scan_siblings(scope.children_scopes());

        // convert any internal returning function
        if (scope.entry()->is_returning() && !scope.entry()->is_external()) {
            a.promote_to_closure();
        }

        // convert any BB that's used as a closure
        else for (auto use : scope.entry()->uses()) {
            if (!is_acceptable_use_for_bb(use))
                a.promote_to_closure();
        }

        src().DLOG("Needs conversion: {} = {}", scope.entry(), a.convert_to_closure);

        if (scope.entry()->is_external()) {
            assert(!a.convert_to_closure);
        }
    }

    DefSet spillable_free_defs(Continuation* entry, DefSet& result, DefSet& rematerialize) {
        unique_queue<DefSet> queue;

        auto& scope = forest_.get_scope(entry);

        for (auto def: scope.free_frontier())
            queue.push(def);
        //entry->world().DLOG("Computing free variables for {}", entry);

        while (!queue.empty()) {
            auto free = queue.pop();
            assert(!free->type()->isa<MemType>());

            if (free == entry)
                continue;

            if (auto tuple = free->isa<Tuple>()) {
                // always rematerialize tuples
                rematerialize.insert(tuple);
                for (auto elem : tuple->ops())
                    queue.push(elem);
            } else if (auto cont = free->isa_nom<Continuation>()) {
                // only capture basic blocks - not top level fns
                // note: capturing basic blocks is bad
                if (!scope.contains(cont)) {
                    auto& other_scope = forest_.get_scope(cont);
                    // top-level continuations don't have parent scopes
                    bool top_level = other_scope.parent_scope() == nullptr;
                    if (!top_level) {
                        //entry->world().DLOG("fv (cont) of {}: {} : {}", entry, cont, cont->type());
                        result.insert(free);
                        //lookup(cont).convert = true;
                    } else {
                        //entry->world().DLOG("ignoring {} because it is top level", free);
                    }
                } else {
                    //entry->world().DLOG("ignoring {} because it's in scope of the lifted cont", free);
                }
            } else if (free->has_dep(Dep::Param) || free->has_dep(Dep::Cont)) {
                //entry->world().DLOG("fv of {}: {} : {}", entry, free, free->type());
                result.insert(free);
            } else {
                //free->world().DLOG("ignoring {} because it has no Param dependency", free);
            }
        }

        //entry->world().DLOG("Computed free variables for {, }", result);

        return result;
    }

    struct ScopeRewriter : public Rewriter {
        ScopeRewriter(ClosureConverter& converter, Scope* scope, ScopeRewriter* parent) : Rewriter(converter.src(), converter.dst()), converter_(converter), parent_(parent), scope_(scope), name_(scope ? scope->entry()->unique_name() : "root") {
            if (parent)
                depth_ = parent->depth_ + 1;
            if (scope)
                analysis_ = &converter_.lookup(scope->entry());
        }

        int depth_ = 0;
        ClosureConverter& converter_;
        ScopeRewriter* parent_;
        Scope* scope_;
        ScopeAnalysis* analysis_;
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

    std::tuple<const Type*, bool> get_env_type(ArrayRef<const Def*> free_vars) {
        // get the environment type
        const Type* env_type = nullptr;
        bool thin_env = free_vars.size() == 1 && is_thin(free_vars[0]->type());

        Array<const Type*> env_ops(free_vars.size());
        for (size_t i = 0, e = free_vars.size(); i != e; ++i)
            env_ops[i] = free_vars[i]->type();
        env_type = src().tuple_type(env_ops);
        return std::make_tuple(env_type, thin_env);
    }

    std::vector<const Type*> rewrite_param_types(ScopeRewriter& rewriter, Continuation* ocont) {
        std::vector<const Type*> nparam_types;

        bool is_accelerator = ocont->is_accelerator();
        bool acc_body_found = false;
        for (auto pt : ocont->type()->types()) {
            const Type* npt = rewriter.instantiate(pt)->as<Type>();
            // in intrinsics, don't closure-convert immediate parameters
            if (ocont->is_intrinsic() && (!is_accelerator || !acc_body_found)) {
                if (pt->tag() == Node_FnType) {
                    npt = dst().fn_type(npt->as<FnType>()->types());
                    if (is_accelerator)
                        acc_body_found = true;
                } else if (auto tuple_t = pt->isa<TupleType>(); tuple_t && tuple_t->types().size() == 2) {
                    // this is because of how we encode match() ...
                    if (tuple_t->types()[1]->tag() == Node_FnType) {
                        auto ntt = npt->as<TupleType>();
                        npt = dst().tuple_type({ntt->types()[0], dst().fn_type(ntt->types()[1]->as<FnType>()->types())});
                    }
                }
            }
            // leave old return params alone
            if (pt->isa<ReturnType>())
                npt = dst().return_type(npt->as<ClosureType>()->types());
            nparam_types.push_back(npt);
        }

        return nparam_types;
    }

    Continuation* as_continuation(const Def* ndef) {
        if (auto found = as_continuations_.lookup(ndef))
            return found.value();

        if (auto ncont = ndef->isa_nom<Continuation>())
            return ncont;
        Debug wr_dbg = ndef->debug();
        auto wrapper = dst().continuation(dst().fn_type(ndef->type()->as<FnType>()->types()), wr_dbg);
        wrapper->jump(ndef, wrapper->params_as_defs());
        as_continuations_[ndef] = wrapper;
        return wrapper;
    }

    World& src_;
    World& dst_;
    World& src() { return src_; }
    World& dst() { return dst_; }

    ScopeRewriter root_rewriter_;
    ScopesForest forest_;
    ContinuationMap<std::unique_ptr<ScopeAnalysis>> analysis_;
    ContinuationMap<std::vector<const Def*>> lifted_env_;
    DefMap<Continuation*> as_continuations_;
    std::vector<std::function<void()>> todo_;

    std::vector<Continuation*> closure_fns_;

    void validate_all_closure_fns_are_top_level();
};

const Def* ClosureConverter::ScopeRewriter::rewrite(const Def* const odef) {
    assert(&odef->world() == &src());
    if (parent_) {
        if (!scope_->contains(odef) && !(analysis_ && analysis_->rematerialize.contains(odef))) {
            //dst().DLOG("Deferring rewriting of {} to {}", odef, parent_->name_);
            return parent_->instantiate(odef);
        }
    }

    if (auto fn_type = odef->isa<FnType>()) {
        Array<const Type*> ntypes(fn_type->num_ops(), [&](int i) {
            auto old_param_t = fn_type->op(i)->as<Type>();
            return instantiate(old_param_t)->as<Type>();
        });
        // returning functions stay returning functions
        if (auto ret_param = fn_type->ret_param_index(); ret_param >= 0) {
            ntypes[ret_param] = dst().return_type(ntypes[ret_param]->as<ClosureType>()->types());
        }

        // Turn all functions into closures, we'll undo it where it is specifically OK
        auto ntype = dst().closure_type(ntypes);
        return ntype;
    } else if (auto ocont = odef->isa_nom<Continuation>()) {
        auto& scope = converter_.forest_.get_scope(ocont);
        ScopeAnalysis& a = converter_.lookup(ocont);

        dst().DLOG("processing continuation (lift={}) '{}' in {}", a.convert_to_closure, ocont, dump());

        auto nparam_types = converter_.rewrite_param_types(*this, ocont);
        if (!a.convert_to_closure) {
            auto ntype = dst().fn_type(nparam_types);
            auto ndef = ocont->stub(*this, ntype);
            insert(odef, ndef);
            ndef->rebuild_from(*this, odef);
            return ndef;
        }

        ScopeRewriter* body_rewriter;
        const Def* ndef = nullptr;
        auto ncont = dst().continuation(dst().fn_type(nparam_types), ocont->attributes(), ocont->debug());

        // make sure to rewrite everything in this scope in a fresh rewriter
        children_.emplace_back(std::make_unique<ScopeRewriter>(converter_, &scope, this));
        body_rewriter = children_.back().get();

        // Compute all the free variables and record additional nodes to be rebuilt in this context
        std::vector<const Def*> free_vars;
        for (auto free : a.free_vars) {
            free_vars.push_back(free);
        }

        Closure* closure = nullptr;
        const Param* closure_param = nullptr;

        auto closure_type = dst().closure_type(nparam_types);

        // add a 'self' parameter to everything we're allowed to mess with
        if (!ocont->is_external()) {
            nparam_types.push_back(closure_type);
            closure_param = ncont->append_param(closure_type, {"self"});
        }

        closure = dst().closure(closure_type, ocont->debug());
        closure->set_fn(ncont, closure_param ? (int) closure_param->index() : -1);
        converter_.closure_fns_.push_back(ncont);
        ndef = closure;
        insert(ocont, closure);
        // what to use as the closure internally
        // if the closure captured something, we want to use the param
        // if it captures nothing, we can use it directly
        const Def* internal_closure = closure;
        if (!free_vars.empty())
            internal_closure = closure_param;
        body_rewriter->insert(ocont, internal_closure);

        converter_.todo_.emplace_back([=]() {
            const Def* new_mem = ncont->mem_param();
            auto [env_type, thin] = converter_.get_env_type(free_vars);
            env_type = this->instantiate(env_type)->as<Type>();

            //dst().DLOG("Old free variables for {} = {, }", ocont, free_vars);

            Array<const Def*> instantiated_free_vars = Array<const Def*>(free_vars.size(), [&](const int i) -> const Def* {
                auto expected_fv_type = instantiate(free_vars[i]->type());
                auto env = instantiate(free_vars[i]);
                if (env->type()->isa<ReturnType>())
                    env = dst().capture_return(env, env->debug());
                assert(env->type() == expected_fv_type);
                //dst().DLOG("Instantiated free variable {}:{} as {}:{}", free_vars[i], free_vars[i]->type(), env, env->type());
                // it cannot be a basic block, and it cannot be top-level either
                // if it used to be a continuation it should be a closure now.
                assert(!env->isa_nom<Continuation>());
                return env;
            });

            dst().DLOG("Instantiated environment variables for {} = {, }", ocont, instantiated_free_vars);

            auto env_tuple = dst().tuple(instantiated_free_vars);
            closure->set_env(env_tuple);
            dst().DLOG("environment of '{}' rewritten as '{}' is {}: {}", ocont, ncont, env_tuple, env_tuple->type());
            //closure->set_env(dst().heap_cell(env_tuple));
            //dst().wdef(ncont, "closure '{}' is leaking memory, type '{}' is too large and must be heap allocated", closure, closure->env()->type());

            assert(closure_param);
            // make the wrapper load the pointer and pass each
            // variable of the environment to the lifted continuation
            auto loaded_env = dst().closure_env(env_type, ncont->mem_param(), closure_param);
            auto env_data = dst().extract(loaded_env, 1_u32);
            new_mem = dst().extract(loaded_env, 0_u32);
            if (free_vars.size() == 1) {
                auto captured = env_data;
                captured->set_name(free_vars[0]->name() + "_captured");
                body_rewriter->insert(free_vars[0], captured);
            } else {
                for (size_t i = 0, e = free_vars.size(); i != e; ++i) {
                    auto captured = dst().extract(env_data, i);
                    captured->set_name(free_vars[i]->name() + "_captured");
                    body_rewriter->insert(free_vars[i], captured);
                }
            }

            // Map old params to new ones
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

            dst().DLOG("finished body of {}", ncont);
        });


        assert(ndef);
        return ndef;
    } else if (auto ret_pt = odef->isa<ReturnPoint>()) {
        auto new_ret_pt = dst().return_point(converter_.as_continuation(instantiate(ret_pt->continuation())));
        return dst().capture_return(new_ret_pt);
    } else if (auto closure = odef->isa<Closure>()) {
        assert(false);
    } else if (auto app = odef->isa<App>()) {
        auto ncallee = instantiate(app->callee());

        std::vector<const Def*> nargs;
        nargs.resize(app->num_args());

        auto ret_param_i = app->callee_type()->ret_param_index();
        for (size_t i = 0; i < app->num_args(); i++) {
            auto oarg = app->arg(i);
            nargs[i] = instantiate(oarg);

            // ensure return params still get a RetType
            if ((int)i == ret_param_i) {
                nargs[i] = dst().return_point(converter_.as_continuation(nargs[i]));
            }
        }

        if (auto ncont = ncallee->isa_nom<Continuation>()) {
            for (size_t i = 0; i < app->num_args(); i++) {
                // ensure the BB arguments to intrinsics such as branch and match are left as continuations
                if (ncont->is_intrinsic()) {
                    auto pt = ncont->type()->types()[i];
                    if (pt->tag() == Node_FnType)
                        nargs[i] = converter_.as_continuation(nargs[i]);
                    else if (auto tuple_t = pt->isa<TupleType>(); tuple_t && tuple_t->types()[1]->tag() == Node_FnType) {
                        nargs[i] = dst().tuple({dst().extract(nargs[i], (u32) 0), converter_.as_continuation(dst().extract(nargs[i], (u32) 1)) });
                    }
                }
            };

            if (ncont->is_accelerator()) {
                // there is unfortunately no standardisation on which parameter is the body parameter for accelerators
                // so we're going to play games here: let's assume only one body is a returning continuation
                // TODO: make accelerator signatures more consistent
                Continuation* body = nullptr;
                size_t body_i;
                for (size_t i = 0; i < app->num_args(); i++) {
                    if (auto cont = nargs[i]->isa_nom<Continuation>(); cont && cont->type()->is_returning()) {
                        assert(!body);
                        body = cont;
                        body_i = i;
                    }
                }
                assert(body);
                auto lifted_body_params = converter_.lifted_env_.lookup(body);

                if (auto extra_params = converter_.lifted_env_.lookup(body); extra_params.has_value()) {
                    // Update the type of the body parameter
                    auto ntypes = ncont->type()->copy_types();
                    ntypes[body_i] = body->type();

                    // Add new parameters to the intrinsic call
                    Continuation* naccelerator = dst().continuation(dst().fn_type(ntypes), ncont->attributes(), ncont->debug());
                    for (auto extra : extra_params.value()) {
                        auto narg = instantiate(extra);
                        nargs.push_back(narg);
                        naccelerator->append_param(narg->type());
                    }
                    ncallee = naccelerator;
                }
            }

            if (auto extra_params = converter_.lifted_env_.lookup(ncont); extra_params.has_value()) {
                for (auto extra : extra_params.value()) {
                    nargs.push_back(instantiate(extra));
                }
            }

        }
        return dst().app(ncallee, nargs, app->debug());
    }
    return Rewriter::rewrite(odef);
}

void ClosureConverter::validate_all_closure_fns_are_top_level() {
    ScopesForest forest(dst());
    for (auto cont : closure_fns_) {
        auto parent = forest.get_scope(cont).parent_scope();
        if (parent != nullptr) {
            dst().ELOG("closure fn {} is not top-level after closure conversion and belongs to {}'s scope", cont, parent);
            assert(false);
        }
    }
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

void lift(Thorin& thorin) {
    auto& src = thorin.world_container();
    auto dst = std::make_unique<World>(*src);
    ClosureConverter converter(*src, *dst);

    converter.scan();

    for (auto& external : src->externals())
        converter.root_rewriter().instantiate(external.second);

    while (!converter.todo_.empty()) {
        auto f = converter.todo_.back();
        converter.todo_.pop_back();
        f();
    }

    converter.validate_all_closure_fns_are_top_level();
    validate_all_returning_functions_top_level(*dst);

    src.swap(dst);
    thorin.cleanup();
}

}
