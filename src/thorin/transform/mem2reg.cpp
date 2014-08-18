#include "thorin/memop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/analyses/verify.h"

namespace thorin {

void mem2reg(const Scope& scope) {
    auto schedule = schedule_late(scope);
    DefMap<size_t> addresses;
    LambdaSet set;
    size_t cur_handle = 0;

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
        // Search for slots/loads/stores from top to bottom and use set_value/get_value to install parameters.
        for (auto primop : schedule[lambda]) {
            auto def = Def(primop);
            if (auto slot = def->isa<Slot>()) {
                // evil HACK
                if (slot->name == "sum_xxx") {
                    addresses[slot] = size_t(-1);     // mark as "address taken"
                    goto next_primop;
                }

                // are all users loads and store?
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
                        store->replace(store->mem());
                    }
                }
            } else if (auto load = def->isa<Load>()) {
                if (auto slot = load->ptr()->isa<Slot>()) {
                    if (addresses[slot] != size_t(-1)) {  // if not "address taken"
                        auto type = slot->type().as<PtrType>()->referenced_type();
                        load->replace(lambda->get_value(addresses[slot], type, slot->name.c_str()));
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

    for (auto lambda : scope)
        lambda->clear();        // clean up value numbering table
}

void mem2reg(World& world) {
    // first we need to care about that this situation does not occur:
    //  a:                      b:
    //      A(..., c)               B(..., c)
    //  otherwise mem2reg does not have lambda to place params
    std::vector<Lambda*> todo;

    for (auto lambda : world.lambdas()) {
        if (lambda->is_basicblock()) {
            auto preds = lambda->preds();
            if (preds.size() > 1) {
                for (auto pred : preds) {
                    if (pred->to() != lambda) {
                        todo.push_back(lambda);
                        goto next_lambda;
                    }
                }
            }
        }
next_lambda:;
    }

    for (auto lambda : todo) {
        for (auto pred : lambda->preds()) {
            // create new lambda
            Type2Type map;
            auto resolver = lambda->stub(map, lambda->name + ".cascading");
            resolver->jump(lambda, resolver->params_as_defs());

            // update pred
            for (size_t i = 0, e = pred->num_args(); i != e; ++i) {
                if (pred->arg(i) == lambda) {
                    pred->update_arg(i, resolver);
                    goto next_pred;
                }
            }
            THORIN_UNREACHABLE;
next_pred:;
        }
    }

    for (auto scope : top_level_scopes(world))
        mem2reg(*scope);

    world.cleanup();
    debug_verify(world);
}

}
