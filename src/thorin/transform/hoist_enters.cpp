#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static void find_enters(std::deque<const Enter*>& enters, const Def* def) {
    if (auto enter = def->isa<Enter>())
        enters.push_front(enter);

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

    // find enters from bottom up
    for (auto n : scope.f_cfg().post_order())
        find_enters(enters, n->continuation());

    if (enters.empty() || enters[0]->mem() != scope.entry()->mem_param())
        return; // do nothing

    auto entry_enter = enters[0];
    auto frame = entry_enter->out_frame();
    enters.pop_front();

    for (auto old_enter : enters) {
        for (auto use : old_enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            slot->replace(world.slot(slot->alloced_type(), frame, slot->loc(), slot->name));
            assert(slot->num_uses() == 0);
        }
    }

    for (auto old_enter : enters)
        old_enter->out_mem()->replace(old_enter->mem());

    if (frame->num_uses() == 0)
        entry_enter->out_mem()->replace(entry_enter->mem());
}

void hoist_enters(World& world) {
    world.cleanup();
    Scope::for_each(world, [] (const Scope& scope) { hoist_enters(scope); });
    world.cleanup();
    debug_verify(world);
}

}
