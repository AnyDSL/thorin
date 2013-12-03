#include <iostream>
#include <unordered_map>

#include "thorin/memop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    auto schedule = schedule_late(scope);
    DefMap<size_t> addresses;
    LambdaSet pass;
    size_t cur_handle = 1; // use 0 for mem
    World& world = scope.world();

    for (Lambda* lambda : scope) {
        for (auto pred : lambda->preds()) {
            if (!scope.contains(pred)) {
#ifndef NDEBUG
                bool found = false;
#endif
                for (auto param : lambda->params()) {
                    if (param->type()->isa<Mem>()) {
                        lambda->set_mem(param);
#ifndef NDEBUG
                        assert(!found);
                        found = true;
#else
                        break;
#endif
                    }
                }
                break;
            }
        }

        // Search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
        for (auto primop : schedule[lambda]) {
            auto def = Def(primop);

            if (auto slot = def->isa<Slot>()) {
                // are all users loads and stores?
                for (auto use : slot->uses()) {
                    if (!use->isa<Load>() && !use->isa<Store>()) {
                        addresses[slot] = size_t(-1);     // mark as "address taken"
                        goto next_primop;
                    }
                }
                addresses[slot] = cur_handle++;
            } else if (auto store = def->isa<Store>()) {
                if (auto slot = store->ptr()->isa<Slot>()) {
                    if (addresses[slot] != size_t(-1)) {  // if not "address taken"
                        lambda->set_value(addresses[slot], store->val());
                        store->replace(lambda->get_mem());
                        continue;
                    }
                }
                store->replace(world.store(lambda->get_mem(), store->ptr(), store->val(), store->name));
                lambda->set_mem(store);
            } else if (auto load = def->isa<Load>()) {
                if (auto slot = load->ptr()->isa<Slot>()) {
                    if (addresses[slot] != size_t(-1)) {  // if not "address taken"
                        auto type = slot->type()->as<Ptr>()->referenced_type();
                        load->extract_val()->replace(lambda->get_value(addresses[slot], type, slot->name.c_str()));
                        load->extract_mem()->replace(lambda->get_mem());
                        continue;
                    }
                }
                auto nload = world.load(lambda->get_mem(), load->ptr(), load->name);
                load->replace(nload);
                lambda->set_mem(nload->extract_mem());
            } else if (auto enter = def->isa<Enter>()) {
                auto nenter = world.enter(lambda->get_mem());
                enter->replace(nenter);
                lambda->set_mem(nenter->extract_mem());
            } else if (auto leave = def->isa<Leave>()) {
                leave->replace(world.leave(lambda->get_mem(), leave->frame()));
                lambda->set_mem(Def(leave));
            }
next_primop:;
        }

        // seal successors of last lambda if applicable
        for (auto succ : lambda->succs()) {
            if (succ->parent() != 0) {
                if (!pass.visit(succ)) {
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
    auto top = top_level_lambdas(world);

    for (auto lambda : world.lambdas()) {   // unseal all lambdas ...
        lambda->set_parent(lambda);
        lambda->unseal();
    }

    for (auto lambda : top) {               // ... except top-level lambdas
        lambda->set_parent(0);
        lambda->seal();
#ifndef NDEBUG
                bool found = false;
#endif
                for (auto param : lambda->params()) {
                    if (param->type()->isa<Mem>()) {
                        lambda->set_mem(param);
#ifndef NDEBUG
                        assert(!found);
                        found = true;
#else
                        break;
#endif
                    }
                }
    }

    for (auto root : top)
        mem2reg(Scope(root));

    world.cleanup();
    debug_verify(world);
}

}
