#include "anydsl2/memop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/placement.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/inliner.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

struct Replace {
    Replace(const Load* load, const Def* with)
        : access_(load)
        , with_(with)
    {}
    Replace(const Store* store)
        : access_(store)
        , with_(0)
    {}

    const Access* access_;
    const Def* with_;
};

void init_lambda(const size_t pass, Lambda* lambda) {
    if (!lambda->visit(pass))
        lambda->counter = lambda->preds().size();
}

void mem2reg(World& world) {
    // mark lambdas passed to other functions as head
    // -> Lambda::get_value will stop at function heads
    for_all (lambda, world.lambdas()) {
        lambda->set_parent(lambda->is_passed() ? 0 : lambda);
        lambda->unseal();
    }

    for_all (root, top_level_lambdas(world)) {
        Scope scope(root);
        std::vector<const Def*> visit = visit_late(scope);
        std::vector<Replace> to_replace;
        const size_t pass = world.new_pass();

        Lambda* cur = 0;
        for_all (def, visit) {
            if (Lambda* lambda = def->isa_lambda()) {
                if (cur) {
                    for_all (succ, cur->succs()) {
                        init_lambda(pass, succ);
                        if (--succ->counter == 0)
                            succ->seal();
                    }
                }
                cur = lambda;
            } else if (const Store* store = def->isa<Store>()) {
                if (const Slot* slot = store->ptr()->isa<Slot>()) {
                    cur->set_value(slot->index(), store->val());
                    to_replace.push_back(Replace(store));
                }
            } else if (const Load* load = def->isa<Load>()) {
                if (const Slot* slot = load->ptr()->isa<Slot>()) {
                    const Type* type = slot->type()->as<Ptr>()->referenced_type();
                    to_replace.push_back(Replace(load, cur->get_value(slot->index(), type, slot->name.c_str())));
                }
            }
        }

        for_all (succ, cur->succs()) {
            assert(succ->is_visited(pass));
            assert(succ->counter == 1);
            succ->seal();
        }

        for (size_t i = to_replace.size(); i-- != 0;) {
            Replace replace = to_replace[i];

            if (const Def* with = replace.with_) {
                const Load* load = replace.access_->as<Load>();
                load->extract_val()->replace(with);
                load->extract_mem()->replace(load->mem());
            } else {
                const Store* store = replace.access_->as<Store>();
                store->replace(store->mem());
            }
        }

        for (size_t i = visit.size(); i-- != 0;) {
            if (const Leave* leave = visit[i]->isa<Leave>()) {
                const Enter* enter = leave->frame()->as<TupleExtract>()->tuple()->as<Enter>();
                if (enter->num_uses() == 2) {
                    enter->extract_mem()->replace(enter->mem());
                    leave->replace(leave->mem());
                }
            }
        }
    }

    for_all (lambda, world.lambdas())
        lambda->clear();

    debug_verify(world);
    world.cleanup();
}

}
