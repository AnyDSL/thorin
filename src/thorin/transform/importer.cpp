#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/primop.h"
#include "partial_evaluation.h"

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
                // avoid messing with external continuations
                if (src().is_external(cont))
                    goto rebuild;

                auto only_called = [&]() -> bool {
                    for (auto use : cont->uses()) {
                        if (use->isa<Param>())
                            continue;
                        if (auto app = use->isa<App>(); app && use.index() == App::Ops::Callee)
                            continue;
                        return false;
                    }
                    return true;
                };

                if (body->args() == cont->params_as_defs()) {
                    // We completely replace the original continuation
                    // If we don't do so, then we miss some simplifications
                    auto ncallee = instantiate(body->callee());

                    // don't rewrite a continuation as a different def, unless it's only ever called
                    if (ncallee->isa<Continuation>() || only_called()) {
                        src().VLOG("simplify: continuation {} calls a free def: {}", cont->unique_name(), body->callee());
                        return ncallee;
                    }
                }

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

                bool has_calls = false;
                // for every use of the continuation at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : cont->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == App::Ops::Callee) {
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

const Def* Importer::find_origin(const Def* ndef) {
    for (auto def : src().defs()) {
        if (ndef == lookup(def))
            return def;
    }
    return nullptr;
}

}
