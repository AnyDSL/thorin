#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/critical_edge_elimination.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    auto schedule = schedule_late(scope);
    DefMap<size_t> addresses;
    LambdaSet set;
    DefSet done;
    size_t cur_handle = 0;

    auto take_address = [&] (const Slot* slot) { addresses[slot] = size_t(-1); };
    auto is_address_taken = [&] (const Slot* slot) { return addresses[slot] == size_t(-1); };

    for (auto lambda : scope)
        lambda->clear_value_numbering_table();

    // unseal all lambdas ...
    for (auto lambda : scope) {
        lambda->set_parent(lambda);
        lambda->unseal();
        assert(lambda->is_cleared());
    }

    // ... except top-level lambdas
    scope.entry()->set_parent(0);
    scope.entry()->seal();

    for (Lambda* lambda : scope) {
        // search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
        for (auto primop : schedule[lambda]) {
            auto def = Def(primop);
            auto p = done.insert(def);
            if (!p.second)
                continue; // already dealt with - we just see this guy again due to replaces
            if (auto slot = def->isa<Slot>()) {
                // evil HACK
                if (slot->name == "sum_xxx") {
                    take_address(slot);
                    goto next_primop;
                }

                // are all users loads and stores *from* this slot (use.index() == 1)?
                for (auto use : slot->uses()) {
                    if (use.index() != 1 || (!use->isa<Load>() && !use->isa<Store>())) {
                        take_address(slot);
                        goto next_primop;
                    }
                }
                addresses[slot] = cur_handle++;
            } else if (auto store = def->isa<Store>()) {
                if (auto slot = store->ptr()->isa<Slot>()) {
                    if (!is_address_taken(slot)) {
                        lambda->set_value(addresses[slot], store->val());
                        store->replace(store->mem());
                    }
                }
            } else if (auto load = def->isa<Load>()) {
                if (auto slot = load->ptr()->isa<Slot>()) {
                    if (!is_address_taken(slot)) {
                        auto type = slot->type().as<PtrType>()->referenced_type();
                        load->out_val()->replace(lambda->get_value(addresses[slot], type, slot->name.c_str()));
                        load->out_mem()->replace(load->mem());
                    }
                }
            }
next_primop:;
        }

        // seal successors of last lambda if applicable
        for (auto succ : scope.succs(lambda)) {
            if (succ->parent() != 0) {
                if (!visit(set, succ)) {
                    assert(addresses.find(succ) == addresses.end());
                    addresses[succ] = succ->preds().size();
                }
                if (--addresses[succ] == 0)
                    succ->seal();
            }
        }
    }
}

void mem2reg(World& world) {
    critical_edge_elimination(world);
    Scope::for_each(world, [] (const Scope& scope) { mem2reg(scope); });
    world.cleanup();
    debug_verify(world);
}

}
