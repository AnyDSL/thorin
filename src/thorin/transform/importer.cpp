#include "thorin/transform/importer.h"

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
        while (auto callee = app->callee()->isa_nom<Continuation>()) {
            if (callee->has_body() && !src().is_external(callee) && callee->can_be_inlined()) {
                todo_ = true;

                for (size_t i = 0; i < callee->num_params(); i++)
                    insert(callee->param(i), import(app->arg(i)));

                app = callee->body();
            } else
                break;
        }

        odef = app;
    } else if (auto cont = odef->isa_nom<Continuation>()) {
        if (cont->has_body()) {
            auto body = cont->body();
            // try to subsume continuations which call a parameter
            // (that is free within that continuation) with that parameter
            if (auto param = body->callee()->isa<Param>()) {
                if (param->continuation() == cont || src().is_external(cont))
                    goto rebuild;

                if (body->args() == cont->params_as_defs()) {
                    insert(odef, instantiate(body->callee()));
                    src().VLOG("simplify: continuation {} calls a parameter  {}", cont->unique_name(), body->filter());
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

                    src().VLOG("simplify: continuation {} calls a parameter (permuted args)", cont->unique_name());
                }

                cont->set_filter(cont->all_true_filter());

                // We just set the filter to true, so this thing gets inlined
                auto rebuilt = Rewriter::rewrite(cont)->as_nom<Continuation>();

                // for every use of the continuation at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : cont->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == App::CALLEE_POSITION) {
                        todo_ = true;
                    }
                }
                return rebuilt;
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
