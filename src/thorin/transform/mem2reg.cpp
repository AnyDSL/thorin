#include <unordered_map>

#include "thorin/memop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    auto schedule = schedule_late(scope);
    const auto pass = scope.world().new_pass();
    size_t cur_handle = 0;

    for (Lambda* lambda : scope) {
        // Search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
        for (auto primop : schedule[lambda]) {
            auto def = Def(primop);
            if (auto slot = def->isa<Slot>()) {
                // are all users loads and store?
                for (auto use : slot->uses()) {
                    if (!use->isa<Load>() && !use->isa<Store>()) {
                        slot->counter = size_t(-1);     // mark as "address taken"
                        goto next_primop;
                    }
                }
                slot->counter = cur_handle++;
            } else if (auto store = def->isa<Store>()) {
                if (auto slot = store->ptr()->isa<Slot>()) {
                    if (slot->counter != size_t(-1)) {  // if not "address taken"
                        lambda->set_value(slot->counter, store->val());
                        store->replace(store->mem());
                    }
                }
            } else if (auto load = def->isa<Load>()) {
                if (auto slot = load->ptr()->isa<Slot>()) {
                    if (slot->counter != size_t(-1)) {  // if not "address taken"
                        auto type = slot->type()->as<Ptr>()->referenced_type();
                        load->extract_val()->replace(lambda->get_value(slot->counter, type, slot->name.c_str()));
                        load->extract_mem()->replace(load->mem());
                    }
                }
            }
next_primop:;
        }

        // seal successors of last lambda if applicable
        for (auto succ : lambda->succs()) {
            if (succ->parent() != 0) {
                if (!succ->visit(pass))
                    succ->counter = succ->preds().size();
                if (--succ->counter == 0)
                    succ->seal();
            }
        }
    }
}

void mem2reg(World& world) {
    auto top = top_level_lambdas(world);

    for (auto lambda : world.lambdas()) {   // unseal all lambdas ...
        lambda->set_parent(lambda);
        lambda->unseal();
    }

    for (auto lambda : top) {               // ... except top-level lambdas
        lambda->set_parent(0);
        lambda->seal();
    }

    for (auto root : top)
        mem2reg(Scope(root));

    world.cleanup();
    debug_verify(world);
}

}
