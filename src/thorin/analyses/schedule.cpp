#include "thorin/analyses/schedule.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

#include <algorithm>
#include <iostream>

namespace thorin {

typedef DefMap<Lambda*> Def2Lambda;

#ifndef NDEBUG
static void verify(const Scope&, const Schedule&) {}
#else
static void verify(const Scope& scope, const Schedule& schedule) {
    auto domtree = scope.domtree();
    LambdaMap<Def> lambda2mem;

    for (auto lambda : scope) {
        Def mem = lambda->mem_param();
        mem = mem ? mem : lambda2mem[domtree->idom(lambda)];
        for (auto primop : schedule[lambda]) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem) {
                    std::cout << "incorrect schedule:" << std::endl;
                    memop->dump();
                    std::cout << "current mem:" << std::endl;
                    mem->dump();
                }

                mem = memop->out_mem();
            }
        }
        lambda2mem[lambda] = mem;
    }
}
#endif

static Def2Lambda schedule_early(const Scope& scope) {
    Def2Lambda def2early;
    DefMap<int> def2num;
    std::queue<Def> queue;

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto op : primop->ops()) {
                if (scope.contains(op))
                    ++num;
            }
            def2num[def] = num;
        }
    }

    auto enqueue_uses = [&] (Def def) {
        for (auto use : def->uses()) {
            if (auto primop = use->isa<PrimOp>()) {
                if (scope.contains(primop)) {
                    if (--def2num[primop] == 0)
                        queue.push(primop);
                }
            }
        }
    };

    for (auto lambda : scope)
        enqueue_uses(lambda);

    for (auto lambda : scope) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                queue.push(param);
        }

        while (!queue.empty()) {
            auto def = pop(queue);
            if (auto primop = def->isa<PrimOp>())
                def2early[primop] = lambda;
            enqueue_uses(def);
        }
    }

    return def2early;
}

const Schedule schedule_late(const Scope& scope) {
    Def2Lambda def2late;
    DefMap<int> def2num;
    auto domtree = scope.domtree();
    std::queue<Def> queue;
    Schedule schedule(scope);

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto use : primop->uses()) {
                if (scope.contains(use))
                    ++num;
            }
            assert(num != 0 && "primop dead");
            def2num[def] += num;
        }
    }

    auto enqueue = [&] (Lambda* lambda, Def def) {
        if (!scope.contains(def) || def->isa_lambda() || def->isa<Param>())
            return;
        auto& late = def2late[def];
        late = late ? domtree->lca(late, lambda) : lambda;
        assert(def2num[def] != 0);
        if (--def2num[def] == 0) {
            queue.push(def);
            if (auto primop = def->isa<PrimOp>())
                schedule.lookup(late).push_back(primop);
        }
    };

    for (auto lambda : scope) {
        for (auto op : lambda->ops())
            enqueue(lambda, op);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        auto lambda = def2late[def];
        for (auto op : def->ops())
            enqueue(lambda, op);
    }

    for (auto& block : schedule.blocks_)
        std::reverse(block.begin(), block.end());

    verify(scope, schedule);
    return schedule;
}

const Schedule schedule_smart(const Scope& scope) {
    Schedule smart(scope);
    auto domtree = scope.domtree();
    auto looptree = scope.looptree();
    auto def2early = schedule_early(scope);
    auto late = schedule_late(scope);

    for (auto lambda : scope) {
        for (auto primop : late[lambda]) {
            assert(scope.contains(primop));
            auto lambda_early = def2early[primop];
            assert(lambda_early != nullptr);
            auto lambda_best = lambda;
            int depth = looptree->depth(lambda_best);
            for (auto i = lambda_best; i != lambda_early;) {
                i = domtree->idom(i);
                int cur_depth = looptree->depth(i);
                if (cur_depth < depth) {
                    lambda_best = i;
                    depth = cur_depth;
                }
            }
            smart.lookup(lambda_best).push_back(primop);
        }
    }

    verify(scope, smart);
    return smart;
}

}
