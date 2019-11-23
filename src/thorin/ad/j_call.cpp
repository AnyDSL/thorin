#include <thorin/ad/j_call.h>
#include <thorin/world.h>

namespace thorin {

const Def* pullback_fn(const Def* fn) {
    THORIN_BREAK;
    if (auto app = fn->isa<App>()) {
        if (auto axiom = app->op(0)->isa<Axiom>()) {
            switch (axiom->tag()) {
            case Tag::ROp: {
            }
            }
        }
    }

    return nullptr;
}

const Def* j_call(const Def* fn) {
    auto& world = fn->world();

    if (auto B = pullback_fn(fn)) {
        (void)B;
        (void)world;
    }

    return nullptr;
}

}

