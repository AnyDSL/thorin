#include <unordered_map>

#include "anydsl2/memop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

void mem2reg(World& world) {
    auto top = top_level_lambdas(world);
    std::vector<Def> enters, leaves;

    for (auto lambda : world.lambdas()) {   // unseal all lambdas ...
        lambda->set_parent(lambda);
        lambda->unseal();
    }

    for (auto lambda : top) {               // ... except top-level lambdas
        lambda->set_parent(0);
        lambda->seal();
    }

    for (auto root : top) {
        Scope scope(root);
        Schedule schedule = schedule_late(scope);
        const size_t pass = world.new_pass();
        size_t cur_handle = 0;
        std::unordered_map<const Load*, Def> load2def;

        for (size_t i = 0, e = scope.size(); i != e; ++i) {
            Lambda* lambda = scope[i];

            // skip lambdas that are connected to higher-order built-ins
            if (lambda->is_connected_to_builtin())
                continue;

            // Search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
            // Then, we know what must be replaced but do not yet replace anything:
            // Defs in the schedule might get invalid!
            for (auto primop : schedule[i]) {
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
                } else if (auto enter = def->isa<Enter>()) {
                    enters.push_back(enter);                // keep track of Enters - they might get superfluous 
                } else if (auto leave = def->isa<Leave>()) {
                    leaves.push_back(leave);                // keep track of Leaves - they might get superfluous 
                }
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
next_primop:;
    }

    // are there any superfluous Leave/Enter pairs?
    for (size_t i = leaves.size(); i-- != 0;) {
        if (auto leave = leaves[i]->isa<Leave>()) {
            auto enter = leave->frame()->as<TupleExtract>()->tuple()->as<Enter>();

            for (auto use : enter->uses()) {
                if (use->isa<Slot>())
                    goto next_leave;
            }

            enter->extract_mem()->replace(enter->mem());
            leave->replace(leave->mem());
        }

next_leave:;
    }

    // are there superfluous poor, lonely Enters? no mercy - eliminate them
    for (size_t i = enters.size(); i-- != 0;) {
        if (auto enter = enters[i]->isa<Enter>()) {
            if (enter->extract_frame()->num_uses() == 0)
                enter->extract_mem()->replace(enter->mem());
        }
    }

    world.eliminate_params();
    debug_verify(world);
}

}
