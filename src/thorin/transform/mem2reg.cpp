#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/critical_edge_elimination.h"
#include "thorin/util/log.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    const auto& cfg = scope.f_cfg();
    DefMap<size_t> slot2handle;
    ContinuationMap<size_t> continuation2num;
    DefSet done;
    size_t cur_handle = 0;

    auto take_address = [&] (const Slot* slot) { slot2handle[slot] = size_t(-1); };
    auto is_address_taken = [&] (const Slot* slot) { return slot2handle[slot] == size_t(-1); };

    // unseal all continuations ...
    for (auto continuation : scope.top_down()) {
        continuation->set_parent(continuation);
        continuation->unseal();
        assert(continuation->is_cleared());
    }

    // ... except top-level continuations
    scope.entry()->set_parent(nullptr);
    scope.entry()->seal();

    // set parent pointers for functions passed to accelerator
    for (auto continuation : scope.top_down()) {
        if (auto callee = continuation->callee()->isa_continuation()) {
            if (callee->is_accelerator()) {
                for (auto arg : continuation->args()) {
                    if (auto acontinuation = arg->isa_continuation()) {
                        if (!acontinuation->is_basicblock()) {
                            DLOG("{} calls accelerator with {}", continuation, acontinuation);
                            acontinuation->set_parent(continuation);
                        }
                    }
                }
            }
        }
    }

    for (const auto& block : schedule(scope, Schedule::Late)) {
        auto continuation = block.continuation();
        // search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters
        for (auto primop : block) {
            if (!done.contains(primop)) {
                if (auto slot = primop->isa<Slot>()) {
                    // are all users loads and stores *from* this slot (use.index() == 1)?
                    for (auto use : slot->uses()) {
                        if (use.index() != 1 || (!use->isa<Load>() && !use->isa<Store>())) {
                            take_address(slot);
                            goto next_primop;
                        }
                    }
                    slot2handle[slot] = cur_handle++;
                } else if (auto store = primop->isa<Store>()) {
                    if (auto slot = store->ptr()->isa<Slot>()) {
                        if (!is_address_taken(slot)) {
                            continuation->set_value(slot2handle[slot], store->val());
                            done.insert(store);
                            store->replace(store->mem());
                        }
                    }
                } else if (auto load = primop->isa<Load>()) {
                    if (auto slot = load->ptr()->isa<Slot>()) {
                        if (!is_address_taken(slot)) {
                            auto type = slot->type()->as<PtrType>()->pointee();
                            auto out_val = load->out_val();
                            auto out_mem = load->out_mem();
                            done.insert(out_val);
                            done.insert(out_mem);
                            out_val->replace(continuation->get_value(slot2handle[slot], type, slot->debug()));
                            out_mem->replace(load->mem());
                        }
                    }
                }
            }
next_primop:;
        }

        // seal successors of last continuation if applicable
        for (auto succ : cfg.succs(block.node())) {
            auto lsucc = succ->continuation();
            if (lsucc->parent() != nullptr) {
                auto i = continuation2num.find(lsucc);
                if (i == continuation2num.end())
                    i = continuation2num.emplace(lsucc, cfg.num_preds(succ)).first;
                if (--i->second == 0)
                    lsucc->seal();
            }
        }
    }
}

void mem2reg(World& world) {
    critical_edge_elimination(world);
    Scope::for_each(world, [] (const Scope& scope) { mem2reg(scope); });
    clear_value_numbering_table(world);
    world.cleanup();
}

}
