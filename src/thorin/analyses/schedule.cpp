#include <algorithm>
#include <iostream>

#include "thorin/lambda.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

typedef LambdaMap<std::vector<const PrimOp*>> Schedule;
typedef DefMap<Lambda*> Def2Lambda;

static Def2Lambda schedule_early(const Scope& scope) {
    Def2Lambda def2early;
    DefMap<int> def2num;
    std::queue<Def> queue;

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto op : primop->ops()) {
                if (scope.contains(op) && !op->isa_lambda())
                    ++num;
            }
            assert(num != 0 && "in scope but no operands");
            def2num[def] = num;
        }
    }

    auto enqueue = [&] (Def def) {
        queue.push(def);
    };

    for (auto lambda : scope) {
        for (auto param : lambda->params()) {
            if (!param->is_proxy())
                enqueue(param);
        }

        while (!queue.empty()) {
            auto def = pop(queue);
            if (auto primop = def->isa<PrimOp>())
                def2early[primop] = lambda;

            for (auto use : def->uses()) {
                if (auto primop = use->isa<PrimOp>()) {
                    if (scope.contains(primop)) {
                        if (--def2num[primop] == 0)
                            enqueue(primop);
                    }
                }
            }
        }
    }

    return def2early;
}

static Schedule schedule_late(const Scope& scope, const Def2Lambda& def2early) {
    Def2Lambda def2late;
    DefMap<int> def2num;
    const DomTree domtree(scope);
    std::queue<Def> queue;
    Schedule schedule;

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto use : primop->uses()) {
                if (scope.contains(use))
                    ++num;
            }
            assert(num != 0 && "primop dead");
            def2num[def] = num;

            if (primop->isa<Enter>() || primop->isa<Slot>() || primop->isa<Load>())
                def2late[primop] = def2early.find(primop)->second;
        }
    }

    auto enqueue = [&] (Lambda* lambda, Def def) {
        auto& late = def2late[def];
        late = late ? domtree.lca(late, lambda) : lambda;
        if (--def2num[def] == 0) {
            queue.push(def);
            if (auto primop = def->isa<PrimOp>())
                schedule[late].push_back(primop);
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

    for (auto& primops : schedule)
        std::reverse(primops.second.begin(), primops.second.end());

    return schedule;
}

void verify(const Scope& scope, Schedule& schedule) {
    const DomTree domtree(scope);
    LambdaMap<Def> lambda2mem;

    for (auto lambda : scope) {
        Def mem = lambda->mem_param();
        if (!mem)
            mem = lambda2mem[domtree.idom(lambda)];
        for (auto primop : schedule[lambda]) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem) {
                    std::cout << "incorrect schedule:" << std::endl;
                    memop->dump();
                    std::cout << "current mem:" << std::endl;
                    mem->dump();
                }

                if (auto out = memop->out_mem())
                    mem = out;
            }
        }
        lambda2mem[lambda] = mem;
    }
}

Schedule schedule_late(const Scope& scope) {
    auto def2early = schedule_early(scope);
    auto schedule = schedule_late(scope, def2early);
    verify(scope, schedule);
    return schedule;
}

Schedule schedule_smart(const Scope& scope) {
    Schedule smart;
    const DomTree domtree(scope); // TODO cache domtree across schedule_late
    const LoopTree looptree(scope);
    auto def2early = schedule_early(scope);
    auto late = schedule_late(scope, def2early);

    for (auto lambda : scope) {
        for (auto primop : late[lambda]) {
            auto lambda_early = def2early[primop];
            auto lambda_best = lambda;
            int depth = looptree.depth(lambda_best);
            for (auto i = lambda_best; i != lambda_early;) {
                i = domtree.idom(i);
                int cur_depth = looptree.depth(i);
                if (cur_depth < depth) {
                    lambda_best = i;
                    depth = cur_depth;
                }
            }
            smart[lambda_best].push_back(primop);
        }
    }

    verify(scope, smart);
    return smart;
}

}
