#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/primop.h"

namespace thorin {

const Def* Importer::rewrite(const Def* const odef) {
    assert(&odef->world() == &src());
    if (auto memop = odef->isa<MemOp>()) {
        // Optimise out dead loads when importing
        if (memop->isa<Load>() || memop->isa<Enter>()) {
            if (memop->out(1)->num_uses() == 0) {
                auto imported_mem = import(memop->mem());
                auto imported_ty = import(memop->out(1)->type())->as<Type>();
                todo_ = true;
                return(dst().tuple({ imported_mem, dst().bottom(imported_ty) }));
            }
        }
    } else if (auto app = odef->isa<App>()) {
        // eat calls to known continuations that are only used once
        if (auto callee = app->callee()->isa_nom<Continuation>()) {
            if (callee->has_body() && !src().is_external(callee) && callee->can_be_inlined()) {
                todo_ = true;
                src().VLOG("simplify: inlining continuation {} because it is called exactly once", callee);
                for (size_t i = 0; i < callee->num_params(); i++)
                    insert(callee->param(i), import(app->arg(i)));

                return instantiate(callee->body());
            } else if (callee->intrinsic() == Intrinsic::Control) {
                auto obody = app->arg(1)->as_nom<Continuation>();
                if (obody->body()->callee() == obody->param(1)) {
                    src().VLOG("simplify: control body just calls the join token, eliminating...");
                    auto mangled_body = drop(obody, {app->arg(0), app->arg(2)});
                    return instantiate(mangled_body->body());
                }
            }
        }
    } else if (auto closure = odef->isa<Closure>()) {
        bool only_called = true;
        for (auto use : closure->uses()) {
            if (use.def()->isa<App>() && use.index() == App::CALLEE_POSITION)
                continue;
            only_called = false;
            break;
        }
        if (only_called) {
            bool self_param_ok = true;
            for (auto use: closure->fn()->params().back()->uses()) {
                // the closure argument can be used, but only to extract the environment!
                if (auto extract = use.def()->isa<Extract>(); extract && is_primlit(extract->index(), 1))
                    continue;
                self_param_ok = false;
                break;
            }
            if (self_param_ok) {
                src().VLOG("simplify: eliminating closure {} as it is never passed as an argument, and is not recursive", closure);
                Array<const Def*> args(closure->fn()->num_params());
                args.back() = closure;
                todo_ = true;
                return instantiate(drop(closure->fn(), args));
            }
        }
    } else if (auto cont = odef->isa_nom<Continuation>()) {
        if (cont->has_body()) {
            auto body = cont->body();
            // try to subsume continuations which call a def
            // (that is free within that continuation) with that def
            auto callee = body->callee();
            auto& scope = forest_->get_scope(cont);
            if (!scope.contains(callee)) {
                if (src().is_external(cont) || callee->type()->tag() != Node_FnType)
                    goto rebuild;

                if (body->args() == cont->params_as_defs()) {
                    src().VLOG("simplify: continuation {} calls a free def: {}", cont->unique_name(), body->callee());
                    // We completely replace the original continuation
                    // If we don't do so, then we miss some simplifications
                    return instantiate(body->callee());
                } else {
                    // build the permutation of the arguments
                    Array<size_t> perm(body->num_args());
                    bool is_permutation = true;
                    for (size_t i = 0, e = body->num_args(); i != e; ++i) {
                        auto param_it = std::find(cont->params().begin(),
                                                  cont->params().end(),
                                                  body->arg(i));

                        if (param_it == cont->params().end()) {
                            is_permutation = false;
                            break;
                        }

                        perm[i] = param_it - cont->params().begin();
                    }

                    if (!is_permutation)
                        goto rebuild;
                }

                bool has_calls = false;
                // for every use of the continuation at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : cont->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == App::CALLEE_POSITION) {
                        todo_ = true;
                        has_calls = true;
                        break;
                    }
                }

                if (has_calls) {
                    auto rebuilt = cont->stub(*this, instantiate(cont->type())->as<Type>());
                    src().VLOG("simplify: continuation {} calls a free def: {} (with permuted args), introducing a wrapper: {}", cont->unique_name(), body->callee(), rebuilt);
                    auto wrapped = dst().run(rebuilt);
                    insert(odef, wrapped);

                    rebuilt->set_body(instantiate(body)->as<App>());
                    return wrapped;
                }
            }
        }
    } else if (auto ret_pt = odef->isa<ReturnPoint>()) {
        auto ret_cont = ret_pt->continuation();
        assert(ret_cont->has_body());
        auto ret_app = ret_cont->body();
        if (ret_app->callee()->type() == ret_pt->type()) {
            bool scopes_ok = true;
            auto p = ret_app->callee()->isa<Param>();
            scopes_ok &= !p || p->continuation() != ret_cont;
            if (scopes_ok && ret_app->args() == ret_cont->params_as_defs()) {
                src().VLOG("simplify: return point {} just forwards data to another: {}", ret_pt, ret_app->callee());
                return instantiate(ret_cont->body()->callee());
            }
        }
    }

    rebuild:
    auto ndef = Rewriter::rewrite(odef);
    if (odef->isa_structural()) {
        // If some substitution took place
        // TODO: this might be dead code at the moment
        todo_ |= odef->tag() != ndef->tag();
    }
    return ndef;
}

}
