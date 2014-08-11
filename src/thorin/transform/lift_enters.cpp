#include "thorin/memop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static const Enter* find_enter(Lambda* lambda) {
    for (auto param : lambda->params()) {
        if (param->type().isa<MemType>()) {
            for (auto use : param->uses()) {
                if (auto enter = use->isa<Enter>())
                    return enter;
            }
        }
    }
    return nullptr;
}

static void lift_enters(const Scope& scope) {
    if (scope.size() <= 1)
        return;

    World& world = scope.world();
    std::vector<const Enter*> enters;

    for (size_t i = scope.size(); i-- != 1;) {
        if (auto enter = find_enter(scope.rpo(i)))
            enters.push_back(enter);
    }

    auto enter = find_enter(scope.entry());
    if (enter == nullptr) {
        for (auto param : scope.entry()->params()) {
            if (param->type().isa<MemType>()) {
                enter = world.enter(scope.entry()->param(0));
                break;
            }
        }
    }
    assert(enter != nullptr);

    // find max slot index
    size_t index = 0;
    for (auto use : enter->uses()) {
        if (auto slot = use->isa<Slot>())
            index = std::max(index, slot->index());
    }

    for (auto old_enter : enters) {
        for (auto use : old_enter->uses()) {
            if (auto slot = use->isa<Slot>())
                slot->replace(world.slot(slot->ptr_type()->referenced_type(), enter, index++, slot->name));
        }
    }
}

void lift_enters(World& world) {
    for (auto scope : top_level_scopes(world))
        lift_enters(*scope);
    world.cleanup();
    debug_verify(world);
}

}
