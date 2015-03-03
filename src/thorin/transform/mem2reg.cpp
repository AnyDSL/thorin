#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/critical_edge_elimination.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    auto& cfg = *scope.f_cfg();
    auto schedule = schedule_late(scope);
    DefMap<size_t> slot2handle;
    LambdaMap<size_t> lambda2num;
    size_t cur_handle = 0;

    auto take_address = [&] (const Slot* slot) { slot2handle[slot] = size_t(-1); };
    auto is_address_taken = [&] (const Slot* slot) { return slot2handle[slot] == size_t(-1); };

    for (auto lambda : scope)
        lambda->clear_value_numbering_table();

    // unseal all lambdas ...
    for (auto lambda : scope) {
        lambda->set_parent(lambda);
        lambda->unseal();
        assert(lambda->is_cleared());
    }

    // ... except top-level lambdas
    scope.entry()->set_parent(nullptr);
    scope.entry()->seal();

    for (auto n : cfg.rpo()) {
        if (auto in = n->isa<InCFNode>()) {
            auto lambda = in->lambda();
            // search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
            for (auto primop : schedule[lambda]) {
                auto def = Def(primop);
                if (auto slot = def->isa<Slot>()) {
                    // are all users loads and store?
                    for (auto use : slot->uses()) {
                        if (!use->isa<Load>() && !use->isa<Store>()) {
                            take_address(slot);
                            goto next_primop;
                        }
                    }
                    slot2handle[slot] = cur_handle++;
                } else if (auto store = def->isa<Store>()) {
                    if (auto slot = store->ptr()->isa<Slot>()) {
                        if (!is_address_taken(slot)) {
                            lambda->set_value(slot2handle[slot], store->val());
                            store->replace(store->mem());
                        }
                    }
                } else if (auto load = def->isa<Load>()) {
                    if (auto slot = load->ptr()->isa<Slot>()) {
                        if (!is_address_taken(slot)) {
                            auto type = slot->type().as<PtrType>()->referenced_type();
                            load->out_val()->replace(lambda->get_value(slot2handle[slot], type, slot->name.c_str()));
                            load->out_mem()->replace(load->mem());
                        }
                    }
                }
next_primop:;
            }
        }

        // seal successors of last lambda if applicable
        for (auto succ : cfg.succs(n)) {
            if (auto in = succ->isa<InCFNode>()) {
                auto lsucc = in->lambda();
                if (lsucc->parent() != nullptr) {
                    auto i = lambda2num.find(lsucc);
                    if (i == lambda2num.end())
                        i = lambda2num.emplace(lsucc, cfg.num_preds(succ)).first;
                    if (--i->second == 0)
                        lsucc->seal();
                }
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
