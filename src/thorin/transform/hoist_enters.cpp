#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static const Enter* find_enter(const Def* def) {
    for (auto use : def->uses()) {
        if (auto enter = use->isa<Enter>())
            return enter;
    }
    return nullptr;
}

static void find_enters(Continuation* continuation, std::vector<const Enter*>& enters) {
    if (auto param = continuation->mem_param()) {
        for (const Def* cur = param; cur;) {
            if (auto memop = cur->isa<MemOp>())
                cur = memop->out_mem();

            if (auto enter = find_enter(cur))
                enters.push_back(enter);

            const auto& uses = cur->uses();
            cur = nullptr;
            for (auto use : uses) {
                if (auto memop = use->isa<MemOp>()) {
                    cur = memop;
                    break;
                }
            }
        }
    }
}

static void hoist_enters(const Scope& scope) {
    World& world = scope.world();
    std::vector<const Enter*> enters;

    for (auto n : scope.f_cfg().post_order()) {
        if (n != scope.f_cfg().entry())
            find_enters(n->continuation(), enters);
    }

    auto mem_param = scope.entry()->mem_param();
    assert(mem_param->num_uses() == 1);
    auto enter = find_enter(mem_param);

    if (enter == nullptr)
        return; // do nothing

    auto frame = enter->out_frame();

    for (auto old_enter : enters) {
        for (auto use : old_enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            slot->replace(world.slot(slot->alloced_type(), frame, slot->loc(), slot->name));
            assert(slot->num_uses() == 0);
        }
    }
}

void hoist_enters(World& world) {
    world.cleanup();
    Scope::for_each(world, [] (const Scope& scope) { hoist_enters(scope); });
    world.cleanup();
    debug_verify(world);
}

}
