#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/split_slots.h"

namespace thorin {

struct IndexHash {
    static hash_t hash(u32 u) { return u; } // TODO bad hash function
    static bool eq(u32 a, u32 b) { return a == b; }
    static u32 sentinel() { return 0xFFFFFFFF; }
};

static void split(const Slot* slot) {
    auto array_type = slot->alloced_type()->as<DefiniteArrayType>();
    auto dim = array_type->dim();
    auto elem_type = array_type->elem_type();

    HashMap<u32, const Def*, IndexHash> new_slots;
    auto& world = slot->world();

    auto elem_slot = [&] (u32 index) {
        if (!new_slots.contains(index))
            new_slots[index] = world.slot(elem_type, slot->frame(), slot->debug());
        return new_slots[index];
    };

    for (auto use : slot->copy_uses()) {
        if (auto lea = use->isa<LEA>()) {
            lea->replace_uses(elem_slot(lea->index()->as<PrimLit>()->value().get_u32()));
        } else if (auto store = use->isa<Store>()) {
            auto in_mem = store->op(0);
            for (size_t i = 0, e = dim; i != e; ++i) {
                auto elem = world.extract(store->op(2), i, store->debug());
                in_mem = world.store(in_mem, elem_slot(i), elem, store->debug());
            }
            store->replace_uses(in_mem);
        } else if (auto load = use->isa<Load>()) {
            auto in_mem = load->op(0);
            auto array = world.bottom(array_type, load->debug());
            for (size_t i = 0, e = dim; i != e; ++i) {
                auto tuple = world.load(in_mem, elem_slot(i), load->debug());
                auto elem = world.extract(tuple, 1_u32, load->debug());
                in_mem = world.extract(tuple, 0_u32, load->debug());
                array = world.insert(array, i, elem, load->debug());
            }
            load->replace_uses(world.tuple({ in_mem, array }, load->debug()));
        }
    }
}

static bool can_split(const Slot* slot) {
    if (!slot->alloced_type()->isa<DefiniteArrayType>())
        return false;

    // only accept LEAs with constant indices and loads and stores
    for (auto use : slot->uses()) {
        if (auto lea = use->isa<LEA>()) {
            if (!lea->index()->no_dep())
                return false;
        } else if (!use->isa<Store>() && !use->isa<Load>()) {
            return false;
        }
    }

    return true;
}

static bool split_slots(const Scope& scope) {
    bool todo = false;
    // TODO
#if 0
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
#endif
    return todo;
}

void split_slots(Thorin& thorin) {
    bool todo = true;
    while (todo) {
        todo = false;
        Scope::for_each(thorin.world(), [&] (const Scope& scope) { todo |= split_slots(scope); });
        thorin.cleanup();
    }
}

}
