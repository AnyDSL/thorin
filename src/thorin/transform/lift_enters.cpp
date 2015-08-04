#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static const Enter* find_enter(Def def) {
    for (auto use : def->uses()) {
        if (auto enter = use->isa<Enter>())
            return enter;
    }
    return nullptr;
}

static void find_enters(Lambda* lambda, std::vector<const Enter*>& enters) {
    if (auto param = lambda->mem_param()) {
        for (Def cur = param; cur;) {
            if (auto memop = cur->isa<MemOp>())
                cur = memop->out_mem();

            if (auto enter = find_enter(cur))
                enters.push_back(enter);

            auto uses = cur->uses();
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

static void lift_enters(const Scope& scope) {
    World& world = scope.world();
    std::vector<const Enter*> enters;

    for (auto n : scope.f_cfg().reverse_in_rpo()) {
        if (n != scope.f_cfg().entry())
            find_enters(n->lambda(), enters);
    }

    auto mem_param = scope.entry()->mem_param();
    assert(mem_param->num_uses() == 1);
    auto enter = find_enter(mem_param);
    if (enter == nullptr) {
        assert(false && "TODO");
        //enter = world.enter(mem_param)->as<Enter>();
    }
    auto frame = enter->out_frame();

    size_t index = 0; // find max slot index
    for (auto use : frame->uses())
        index = std::max(index, use->as<Slot>()->index());

    for (auto old_enter : enters) {
        for (auto use : old_enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            slot->replace(world.slot(slot->alloced_type(), frame, index++, slot->name));
        }
        assert(!old_enter->is_proxy());
        old_enter->out_mem()->replace(old_enter->mem());
    }

#ifndef NDEBUG
    for (auto old_enter : enters)
        assert(old_enter->out_frame()->num_uses() == 0);
#endif
}

void lift_enters(World& world) {
    world.cleanup();
    Scope::for_each(world, [] (const Scope& scope) { lift_enters(scope); });
    world.cleanup();
    debug_verify(world);
}

}
