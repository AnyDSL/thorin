#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/split_slots.h"
#include "thorin/util/log.h"

namespace thorin {

static void split(const Slot* slot) {
    Def2Def new_slots;
    auto& world = slot->world();

    for (auto use : slot->uses()) {
        auto lea = use->as<LEA>();
        auto index = lea->index();
        if (!new_slots.contains(index))
            new_slots[index] = world.slot(lea->type()->as<PtrType>()->pointee(), slot->frame(), slot->debug());
        lea->replace(new_slots[index]);
    }
}

static bool can_split(const Slot* slot) {
    for (auto use : slot->uses()) {
        auto lea = use->isa<LEA>();
        if (!lea || !is_const(lea->index()))
            return false;
    }
    return true;
}

static bool split_slots(const Scope& scope) {
    bool todo = false;
    for (const auto& block : schedule(scope, Schedule::Late)) {
        for (auto primop : block) {
            if (auto slot = primop->isa<Slot>()) {
                if (can_split(slot)) {
                    split(slot);
                    todo = true;
                }
            }
        }
    }
    return todo;
}

void split_slots(World& world) {
    bool todo = true;
    while (todo) {
        todo = false;
        Scope::for_each(world, [&] (const Scope& scope) { todo = split_slots(scope); });
        world.cleanup();
    }
}

}
