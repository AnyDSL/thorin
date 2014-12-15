#include "thorin/analyses/schedule.h"

#include <iostream>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

typedef DefMap<Lambda*> Def2Lambda;

#ifndef NDEBUG
static void verify(const Scope& scope, const Schedule& schedule) {
    auto& domtree = *scope.cfa()->domtree();
    LambdaMap<Def> lambda2mem;

    for (auto n : *scope.cfa()->f_cfg()) {
        auto lambda = n->lambda();
        Def mem = lambda->mem_param();
        mem = mem ? mem : lambda2mem[domtree.lookup(scope.cfa()->lookup(lambda))->idom()->lambda()];
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
#else
static void verify(const Scope&, const Schedule&) {}
#endif

static Def2Lambda schedule_early(const Scope& scope) {
    Def2Lambda def2early;
    DefMap<int> def2num;
    std::queue<Def> queue;

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

    auto enqueue_uses = [&] (Def def) {
        for (auto use : def->uses()) {
            if (auto primop = use->isa<PrimOp>()) {
                if (scope._contains(primop)) {
                    if (--def2num[primop] == 0)
                        queue.push(primop);
                }
            }
        }
    };

    for (auto n : *scope.cfa()->f_cfg())
        enqueue_uses(n->lambda());

    for (auto n : *scope.cfa()->f_cfg()) {
        auto lambda = n->lambda();
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
    auto cfg = scope.cfa()->f_cfg();
    auto domtree = scope.cfa()->domtree();
    std::queue<Def> queue;
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

    auto enqueue = [&] (Lambda* lambda, Def def) {
        if (!scope._contains(def) || def->isa_lambda() || def->isa<Param>())
            return;
        auto& late = def2late[def];
        late = late ? domtree->lca(
                domtree->lookup(cfg->lookup(late)), 
                domtree->lookup(cfg->lookup(lambda)))->lambda() : lambda;
        assert(def2num[def] != 0);
        if (--def2num[def] == 0) {
            queue.push(def);
            if (auto primop = def->isa<PrimOp>())
                schedule.lookup(late).push_back(primop);
        }
    };

    for (auto n : *scope.cfa()->f_cfg()) {
        auto lambda = n->lambda();
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
    auto& cfg = *scope.cfa()->f_cfg();
    auto domtree = cfg.domtree();
    auto looptree = scope.cfa()->looptree(); // TODO
    auto def2early = schedule_early(scope);
    auto late = schedule_late(scope);

    for (auto n : *scope.cfa()->f_cfg()) {
        auto lambda = n->lambda();
        for (auto primop : late[lambda]) {
            assert(scope._contains(primop));
            auto lambda_early = def2early[primop];
            assert(lambda_early != nullptr);
            auto lambda_best = lambda;

            if (primop->isa<Enter>() || primop->isa<Slot>() || Enter::is_out_mem(primop) || Enter::is_out_frame(primop))
                lambda_best = lambda_early;
            else {
                int depth = looptree->depth(cfg.lookup(lambda_best));
                for (auto i = lambda_best; i != lambda_early;) {
                    i = domtree->lookup(cfg.lookup(i))->idom()->lambda();
                    int cur_depth = looptree->cf_node2leaf(cfg.lookup(i))->depth();
                    if (cur_depth < depth) {
                        lambda_best = i;
                        depth = cur_depth;
                    }
                }
            }
            smart.lookup(lambda_best).push_back(primop);
        }
    }

    verify(scope, smart);
    return smart;
}

}
