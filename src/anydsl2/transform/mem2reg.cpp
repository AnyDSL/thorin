#include "anydsl2/memop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/placement.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/inliner.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

// currently, this transformation only works when in CFF

void mem2reg(World& world) {
    // mark lambdas passed to other functions as head
    // -> Lambda::get_value will stop at function heads
    for_all (lambda, world.lambdas()) {
        lambda->set_parent(lambda->is_passed() ? 0 : lambda);
        lambda->unseal();
    }

    AutoVector<Tracker*> enters;
    AutoVector<Tracker*> leaves;

    for_all (root, top_level_lambdas(world)) {
        Scope scope(root);
        std::vector<const Access*> accesses;
        Places places = visit_late(scope);
        const size_t pass = world.new_pass();
        size_t cur_handle = 0;

        for (size_t i = 0, e = scope.size(); i != e; ++i) {
            Lambda* lambda = scope[i];

            for_all (def, places[i]) {
                if (const Slot* slot = def->isa<Slot>()) {
                    // are all users loads and store?
                    for_all (use, slot->uses()) {
                        if (!use->isa<Load>() && !use->isa<Store>()) {
                            slot->counter = size_t(-1); // mark as "address taken"
                            goto next_def;
                        }
                    }
                    slot->counter = cur_handle++;
                } else if (const Store* store = def->isa<Store>()) {
                    if (const Slot* slot = store->ptr()->isa<Slot>()) {
                        if (slot->counter != size_t(-1)) { // if not "address taken"
                            lambda->set_value(slot->counter, store->val());
                            accesses.push_back(store);
                        }
                    }
                } else if (const Load* load = def->isa<Load>()) {
                    if (const Slot* slot = load->ptr()->isa<Slot>()) {
                        if (slot->counter != size_t(-1)) { // if not "address taken"
                            const Type* type = slot->type()->as<Ptr>()->referenced_type();
                            load->cptr = lambda->get_value(slot->counter, type, slot->name.c_str());
                            accesses.push_back(load);
                        }
                    }
                } else if (const Enter* enter = def->isa<Enter>()) {
                    enters.push_back(new Tracker(enter));
                } else if (const Leave* leave = def->isa<Leave>()) {
                    leaves.push_back(new Tracker(leave));
                }
            }

            // seal successors of last lambda if applicable
            for_all (succ, lambda->succs()) {
                if (!succ->visit(pass))
                    succ->counter = succ->preds().size();
                if (--succ->counter == 0)
                    succ->seal();
            }
        }

        for (size_t i = accesses.size(); i-- != 0;) {
            if (const Load* load = accesses[i]->isa<Load>()) {
                load->extract_val()->replace((const Def*) load->cptr);
                load->extract_mem()->replace(load->mem());
            } else {
                const Store* store = accesses[i]->as<Store>();
                store->replace(store->mem());
            }
        }

next_def:;
    }

    for_all (lambda, world.lambdas())
        lambda->clear();

    world.cleanup();

    for (size_t i = leaves.size(); i-- != 0;) {
        const Leave* leave = leaves[i]->def()->as<Leave>();
        const Enter* enter = leave->frame()->as<TupleExtract>()->tuple()->as<Enter>();

        for_all (use, enter->uses()) {
            if (use->isa<Slot>())
                goto next_leave;
        }

        enter->extract_mem()->replace(enter->mem());
        leave->replace(leave->mem());

next_leave:;
    }

    for (size_t i = enters.size(); i-- != 0;) {
        if (const Enter* enter = enters[i]->def()->isa<Enter>()) {
            if (enter->extract_frame()->num_uses() == 0)
                enter->extract_mem()->replace(enter->mem());
        }
    }

    debug_verify(world);
}

}
