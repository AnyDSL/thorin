#include "thorin/transform/importer.h"

namespace thorin {

const Def* Importer::rewrite(const Def* odef) {
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
