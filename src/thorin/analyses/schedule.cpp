#include <algorithm>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"

#include "thorin/analyses/schedule.h"

namespace thorin {

Schedule schedule_early(const Scope& scope) {
    Schedule schedule;
    std::queue<Def> queue;
    DefMap<size_t> num_placed;
    DefSet set;

    for (Lambda* lambda : scope) {
        auto& primops = schedule[lambda];

        for (auto param : lambda->params())
            if (!param->is_proxy())
                queue.push(param);

        while (!queue.empty()) {
            Def def = queue.front();
            if (auto primop = def->isa<PrimOp>())
                primops.push_back(primop);
            queue.pop();

            for (auto use : def->uses()) {
                if (use->isa<Lambda>())
                    continue;
                if (set.visit(use))
                    --num_placed[use];
                else {
                    num_placed[use] = -1;
                    for (auto op : use->ops()) {
                        if (!op->is_const())
                            ++num_placed[use];
                    }
                }
                assert(num_placed[use] != size_t(-1));

                if (num_placed[use] == 0)
                    queue.push(use);
            }
        }
    }

    return schedule;
}

Schedule schedule_late(const Scope& scope, DefMap<Lambda*> &late_mapping) {
    Schedule schedule;
    const DomTree domtree(scope);
    LambdaMap<std::queue<const PrimOp*>> queues;
    DefMap<size_t> placed_uses;
    late_mapping.clear();

    for (Lambda* cur : scope.backwards_rpo()) {
        auto& queue = queues[cur] = std::queue<const PrimOp*>();

        auto fill_queue = [&] (Def def) {
            for (auto op : def->ops()) {
                if (auto primop = op->is_non_const_primop()) {
                    queue.push(primop);

                    if (!late_mapping.visit(primop, cur)) { // init unseen primops
                        placed_uses[primop] = primop->num_uses() - 1;
                    }
                }
            }
        };

        fill_queue(cur);

        while (!queue.empty()) {
            const PrimOp* primop = queue.front();
            queue.pop();
            assert(late_mapping.contains(primop));

            if (placed_uses[primop] == 0) {
                Lambda*& late = late_mapping[primop];

                if (late == cur) {
                    schedule[late].push_back(primop);
                    fill_queue(primop);
                } else {
                    late = late ? domtree.lca(cur, late) : cur;
                    queues[late].push(primop);
                }
            } else
                --placed_uses[primop];
        }
    }

    for (auto& primops : schedule)
        std::reverse(primops.second.begin(), primops.second.end());

    return schedule;
}

Schedule schedule_smart(const Scope& scope) {
    Schedule smart;
    const DomTree domtree(scope); // TODO cache domtree across schedule_late
    const LoopTree looptree(scope);
    Schedule early = schedule_early(scope);
    DefMap<Lambda*> late_mapping;
    schedule_late(scope, late_mapping); // set late pointers in primop and remember pass

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda_early = scope[i];
        for (auto primop : early[lambda_early]) {
            if (!late_mapping.contains(primop))
                continue;       // primop is dead
            Lambda* lambda_best = late_mapping[primop];
            assert(scope.contains(lambda_best));
            int depth = std::numeric_limits<int>::max();
            for (Lambda* i = lambda_best; i != lambda_early; i = domtree.idom(i)) {
                int cur_depth = looptree.depth(i);
                if (cur_depth < depth) {
                    lambda_best = i;
                    depth = cur_depth;
                }
            }
            smart[lambda_best].push_back(primop);
        }
    }

    return smart;
}

} // namespace thorin
