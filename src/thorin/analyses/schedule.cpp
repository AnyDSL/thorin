#include "thorin/analyses/schedule.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"
#include "thorin/util/queue.h"

namespace thorin {

typedef DefMap<const CFNode*> Def2CFNode;

Schedule::Schedule(const Scope& scope)
    : scope_(scope)
    , indices_(cfg())
    , blocks_(cfa().size())
{
    block_schedule();
}

void Schedule::block_schedule() {
    // until we have sth better simply use the RPO of the CFG
    size_t i = 0;
    for (auto n : cfg().reverse_post_order()) {
        auto& block = blocks_[i];
        block.node_ = n;
        block.index_ = i;
        indices_[n] = i++;
    }
    assert(blocks_.size() == i);
}

void Schedule::verify() {
#ifndef NDEBUG
    auto& domtree = cfg().domtree();
    Schedule::Map<const Def*> block2mem(*this);

    for (auto& block : *this) {
        const Def* mem = block.lambda()->mem_param();
        mem = mem ? mem : block2mem[(*this)[domtree.idom(block.node())]];
        for (auto primop : block) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem)
                    WLOG("incorrect schedule: % (current mem is %)", memop, mem);
                mem = memop->out_mem();
            }
        }
        block2mem[block] = mem;
    }
#endif
}

static Def2CFNode schedule_early(const Scope& scope) {
    Def2CFNode def2early;
    DefMap<int> def2num;
    auto& cfg = scope.f_cfg();
    std::queue<const Def*> queue;

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto op : primop->ops()) {
                if (scope._contains(op))
                    ++num;
            }
            def2num[def] = num;
        }
    }

    auto enqueue_uses = [&] (const Def* def) {
        for (auto use : def->uses()) {
            if (auto primop = use->isa<PrimOp>()) {
                if (scope._contains(primop)) {
                    if (--def2num[primop] == 0)
                        queue.push(primop);
                }
            }
        }
    };

    for (auto n : cfg.reverse_post_order())
        enqueue_uses(n->lambda());

    for (auto n : cfg.reverse_post_order()) {
        for (auto param : n->lambda()->params()) {
            //if (!param->is_proxy())
            // TODO
            queue.push(param);
        }

        while (!queue.empty()) {
            auto def = pop(queue);
            if (auto primop = def->isa<PrimOp>())
                def2early[primop] = n;
            enqueue_uses(def);
        }
    }

    return def2early;
}

Schedule schedule_late(const Scope& scope) {
    Def2CFNode def2late;
    DefMap<int> def2num;
    auto& cfg = scope.f_cfg();
    auto& domtree = cfg.domtree();
    std::queue<const Def*> queue;
    Schedule schedule(scope);

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto use : primop->uses()) {
                if (scope._contains(use))
                    ++num;
            }
            assert(num != 0 && "primop dead");
            def2num[def] += num;
        }
    }

    auto enqueue = [&] (const CFNode* n, const Def* def) {
        if (!scope._contains(def) || def->isa_lambda() || def->isa<Param>())
            return;
        auto& late = def2late[def];
        late = late ? domtree.lca(late, n) : n;
        assert(def2num[def] != 0);
        if (--def2num[def] == 0) {
            queue.push(def);
            if (auto primop = def->isa<PrimOp>())
                schedule.append(late, primop);
        }
    };

    for (auto n : cfg.reverse_post_order()) {
        for (auto op : n->lambda()->ops())
            enqueue(n, op);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        auto lambda = def2late[def];
        for (auto op : def->ops())
            enqueue(lambda, op);
    }

    for (auto& block : schedule.blocks_)
        std::reverse(block.primops_.begin(), block.primops_.end());

    schedule.verify();
    return std::move(schedule);
}

Schedule schedule_smart(const Scope& scope) {
    Schedule smart(scope);
    auto& cfg = scope.f_cfg();
    auto& domtree = cfg.domtree();
    auto& looptree = cfg.looptree();
    auto def2early = schedule_early(scope);
    auto late = schedule_late(scope);

    for (auto& block : late) {
        for (auto primop : block) {
            assert(scope._contains(primop));
            auto node_early = def2early[primop];
            assert(node_early != nullptr);
            auto node_best = block.node();

            if (primop->isa<Enter>() || primop->isa<Slot>() || Enter::is_out_mem(primop) || Enter::is_out_frame(primop))
                node_best = node_early;
            else {
                int depth = looptree[node_best]->depth();
                for (auto i = node_best; i != node_early;) {
                    i = domtree.idom(i);
                    int cur_depth = looptree[i]->depth();
                    if (cur_depth < depth) {
                        node_best = i;
                        depth = cur_depth;
                    }
                }
            }
            smart.append(node_best, primop);
        }
    }

    smart.verify();
    return std::move(smart);
}

}
