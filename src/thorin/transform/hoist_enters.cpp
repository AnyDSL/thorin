#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/nest.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static void find_enters(std::deque<const Enter*>& enters, const Def* def) {
    if (auto enter = def->isa<Enter>())
        enters.push_back(enter);

    if (auto memop = def->isa<MemOp>())
        def = memop->out_mem();

    for (auto use : def->uses()) {
        if (auto memop = use->isa<MemOp>())
            find_enters(enters, memop);
    }
}

static void find_enters(std::deque<const Enter*>& enters, Continuation* continuation) {
    if (auto mem_param = continuation->mem_param())
        find_enters(enters, mem_param);
}

static void hoist_enters(const Scope& scope) {
    World& world = scope.world();
    std::deque<const Enter*> enters;

    for (auto n : scope.nest().top_down())
        find_enters(enters, n->continuation());


    if (enters.empty() || enters[0]->mem() != scope.entry()->mem_param()) {
        VLOG("cannot optimize {} - didn't find entry enter", scope.entry());
        return;
    }

    auto entry_enter = enters[0];
    auto frame = entry_enter->out_frame();
    enters.pop_front();

    for (auto i = enters.rbegin(), e = enters.rend(); i != e; ++i) {
        auto old_enter = *i;
        for (auto use : old_enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            slot->replace(world.slot(slot->alloced_type(), frame, slot->debug()));
            assert(slot->num_uses() == 0);
        }
    }

    for (auto i = enters.rbegin(), e = enters.rend(); i != e; ++i)
        (*i)->out_mem()->replace((*i)->mem());

    if (frame->num_uses() == 0)
        entry_enter->out_mem()->replace(entry_enter->mem());
}

void hoist_enters(World& world) {
    Scope::for_each(world, [] (const Scope& scope) { hoist_enters(scope); });
    world.cleanup();
}

}
