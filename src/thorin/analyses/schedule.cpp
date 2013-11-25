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
                    --use->counter;
                else {
                    use->counter = -1;
                    for (auto op : use->ops()) {
                        if (!op->is_const())
                            ++use->counter;
                    }
                }
                assert(use->counter != size_t(-1));

                if (use->counter == 0)
                    queue.push(use);
            }
        }
    }

    return schedule;
}

static inline Lambda*& get_late(const PrimOp* primop) { return (Lambda*&) primop->ptr; }

Schedule schedule_late(const Scope& scope, DefSet& set) {
    Schedule schedule;
    LambdaMap<std::queue<const PrimOp*>> queues;
    set.clear();

    for (Lambda* cur : scope.backwards_rpo()) {
        auto& queue = queues[cur] = std::queue<const PrimOp*>();

        auto fill_queue = [&] (Def def) {
            for (auto op : def->ops()) {
                if (auto primop = op->is_non_const_primop()) {
                    queue.push(primop);

                    if (!set.visit(primop)) { // init unseen primops
                        get_late(primop) = cur;
                        primop->counter = primop->num_uses() - 1;
                    }
                }
            }
        };

        fill_queue(cur);

        while (!queue.empty()) {
            const PrimOp* primop = queue.front();
            queue.pop();
            assert(set.contains(primop));

            if (primop->counter == 0) {
                Lambda*& late = get_late(primop);

                if (late == cur) {
                    schedule[late].push_back(primop);
                    fill_queue(primop);
                } else {
                    late = late ? scope.domtree().lca(cur, late) : cur;
                    queues[late].push(primop);
                }
            } else
                --primop->counter;
        }
    }

    for (auto& primops : schedule)
        std::reverse(primops.second.begin(), primops.second.end());

    return schedule;
}

Schedule schedule_smart(const Scope& scope) {
    Schedule smart;
    Schedule early = schedule_early(scope);
    DefSet set;
    schedule_late(scope, set); // set late pointers in primop and remember pass

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda_early = scope[i];
        for (auto primop : early[lambda_early]) {
            if (!set.contains(primop))
                continue;       // primop is dead
            Lambda* lambda_best = get_late(primop);
            assert(scope.contains(lambda_best));
            int depth = std::numeric_limits<int>::max();
            for (Lambda* i = lambda_best; i != lambda_early; i = scope.domtree().idom(i)) {
                int cur_depth = scope.looptree().depth(i);
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
