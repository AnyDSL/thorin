#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/primop.h"

namespace thorin {

const Def* Importer::rewrite(const Def* odef) {
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
        while (true) {
            if (auto callee = app->callee()->isa_nom<Continuation>()) {
                if (callee->has_body() && !src().is_external(callee) && callee->can_be_inlined()) {
                    todo_ = true;
                    src().VLOG("simplify: inlining continuation {} because it is called exactly once", callee);
                    for (size_t i = 0; i < callee->num_params(); i++)
                        insert(callee->param(i), import(app->arg(i)));

                    app = callee->body();
                    continue;
                } else if (callee->intrinsic() == Intrinsic::Control) {
                    auto obody = app->arg(1)->as_nom<Continuation>();
                    if (obody->body()->callee() == obody->param(1)) {
                        src().VLOG("simplify: control body just calls the join token, eliminating...");
                        auto mangled_body = drop(obody, {app->arg(0), app->arg(2)});
                        app = mangled_body->body();
                        continue;
                    }
                }
            } else if (auto closure = app->callee()->isa<Closure>()) {
                if (closure->uses().size() == 1) {
                    bool ok = true;
                    for (auto use: closure->fn()->params().back()->uses()) {
                        // the closure argument can be used, but only to extract the environment!
                        if (auto extract = use.def()->isa<Extract>(); extract && is_primlit(extract->index(), 1))
                            continue;
                        ok = false;
                    }
                    if (ok) {
                        src().VLOG("simplify: inlining closure {} as it is used only once, and is not recursive", closure);
                        Array<const Def*> args(closure->fn()->num_params());
                        std::copy(app->args().begin(), app->args().end(), args.begin());
                        args.back() = closure;
                        app = drop(closure->fn(), args)->body();
                        continue;
                    }
                }
            }
            break;
        }

        odef = app;
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

                    src().VLOG("simplify: continuation {} calls a free def: {} (with permuted args)", cont->unique_name(), body->callee());
                }

                auto rebuilt = cont->stub(*this, instantiate(cont->type())->as<Type>());
                dst().VLOG("rebuilt as {}", rebuilt->unique_name());
                auto wrapped = dst().run(rebuilt);
                insert(odef, wrapped);

                rebuilt->set_body(instantiate(body)->as<App>());

                // for every use of the continuation at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : cont->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == App::CALLEE_POSITION) {
                        todo_ = true;
                    }
                }
                return wrapped;
            }
        }
    } else while (auto ret_pt = odef->isa<ReturnPoint>()) {
        auto ret_cont = ret_pt->continuation();
        assert(ret_cont->has_body());
        auto ret_app = ret_cont->body();
        if (ret_app->callee()->type() == ret_pt->type()) {
            bool scopes_ok = true;
            auto p = ret_app->callee()->isa<Param>();
            scopes_ok &= !p || p->continuation() != ret_cont;

            for (auto a : ret_app->args()) {
                if (auto p = a->isa<Param>()) {
                    if (p->continuation() == ret_cont) {
                        // bail out
                        scopes_ok = false;
                        break;
                    }
                }
            }
            if (scopes_ok) {
                src().VLOG("simplify: return point {} just forwards data to another: {}", ret_pt, ret_app->callee());
                odef = ret_cont->body()->callee();
                continue;
            }
        }
        break;
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
