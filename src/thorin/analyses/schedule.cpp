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
            if (num != 0) // in scope but no operands
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
            def2num[def] += num;

            auto memop = primop->isa<MemOp>();
            // schedule Slots and MemOps without memory out always early
            if (primop->isa<Slot>() || (memop && !memop->has_mem_out()))
                def2late[primop] = def2early.find(primop)->second;

            // artificially increase the use counter of all non-memory-out uses of the mem's operand of a memory-out-primop
            if (memop && memop->has_mem_out()) {
                auto mem = memop->mem();
                assert(scope.contains(mem));
                for (auto use : mem->uses()) {
                    if (scope.contains(use)) {
                        if (auto umemop = use->isa<MemOp>()) {
                            if (!umemop->has_mem_out())
                                ++def2num[umemop];
                        }
                    }
                }
            }
        }
    }

    auto enqueue = [&] (Lambda* lambda, Def def) {
        if (!scope.contains(def) || def->isa_lambda() || def->isa<Param>())
            return;
        auto& late = def2late[def];
        late = late ? domtree.lca(late, lambda) : lambda;
        assert(def2num[def] != 0);
        if (--def2num[def] == 0) {
            queue.push(def);
            if (auto primop = def->isa<PrimOp>()) {
                schedule[late].push_back(primop);
                // now repair the artificially increased counter
                if (auto memop = primop->isa<MemOp>()) {
                    if (memop->has_mem_out()) {
                        auto mem = memop->mem();
                        assert(scope.contains(mem));
                        for (auto use : mem->uses()) {
                            if (scope.contains(use)) {
                                if (auto umemop = use->isa<MemOp>()) {
                                    if (!umemop->has_mem_out()) {
                                        assert(def2num[umemop] != 0);
                                        if (--def2num[umemop] == 0) {
                                            queue.push(umemop);
                                            schedule[late].push_back(umemop);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
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

static void verify(const Scope& scope, Schedule& schedule) {
#ifndef NDEBUG
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

                if (auto out = memop->mem_out())
                    mem = out;
            }
        }
        lambda2mem[lambda] = mem;
    }
#endif
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
