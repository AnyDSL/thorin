#include "thorin/transform/importer.h"

namespace thorin {

const Def* Importer::rewrite(const Def* odef) {
    auto ndef = Rewriter::rewrite(odef);
    if (odef->isa_structural()) {
        todo_ |= odef->tag() != ndef->tag();
    }
    return ndef;
}

}
