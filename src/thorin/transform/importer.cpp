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
    }

    auto ndef = Rewriter::rewrite(odef);
    if (odef->isa_structural()) {
        // If some substitution took place
        // TODO: this might be dead code at the moment
        todo_ |= odef->tag() != ndef->tag();
    }
    return ndef;
}

}
