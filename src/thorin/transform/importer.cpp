#include "thorin/transform/importer.h"
#include "mangle.h"

namespace thorin {

const Def* Importer::rewrite(const Def* odef) {
    assert(&odef->world() == &src());
    again:
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

                odef = callee->body();
                goto again;
            } else if (callee->intrinsic() == Intrinsic::Control) {
                if (auto obody = app->arg(1)->isa_nom<Continuation>()) {
                    if (obody->body()->callee() == obody->param(1) && obody->param(1)->uses().size() == 1) {
                        auto dropped = drop(obody, { app->arg(0), nullptr });
                        odef = src().tuple(dropped->body()->args());
                        src().VLOG("simplify: replaced control construct {} by insides {}", app, odef);
                        goto again;
                    }
                }
            }
        }
    } else if (auto cont = odef->isa_nom<Continuation>()) {
        if (cont->has_body()) {
            auto body = cont->body();
            // try to subsume continuations which call a parameter
            // (that is free within that continuation) with that parameter
            if (auto param = body->callee()->isa<Param>()) {
                if (param->continuation() == cont || src().is_external(cont) || param->type()->tag() != Node_FnType)
                    goto rebuild;

                if (body->args() == cont->params_as_defs()) {
                    src().VLOG("simplify: continuation {} calls a parameter: {}", cont->unique_name(), body->callee());
                    // We completely replace the original continuation
                    // If we don't do so, then we miss some simplifications
                    return instantiate(body->callee());
                } else {
                    // build the permutation of the arguments
                    Array<size_t> perm(body->num_args());
                    bool is_permutation = true;
                    for (size_t i = 0, e = body->num_args(); i != e; ++i) {
                        auto param_it = std::find(cont->params().begin(), cont->params().end(), body->arg(i));

                        if (param_it == cont->params().end()) {
                            is_permutation = false;
                            break;
                        }

                        perm[i] = param_it - cont->params().begin();
                    }

                    if (!is_permutation)
                        goto rebuild;

                    src().VLOG("simplify: continuation {} calls a parameter: {} (with permuted args)", cont->unique_name(), body->callee());
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
