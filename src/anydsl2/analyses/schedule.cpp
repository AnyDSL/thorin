#include <algorithm>
#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/memop.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

typedef Array<std::vector<const PrimOp*>> Schedule;

Schedule schedule_early(const Scope& scope) {
    Schedule schedule(scope.size());
    std::queue<Def> queue;
    const auto pass = scope.world().new_pass();

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda = scope[i];
        auto& primops = schedule[i];

        for (auto param : lambda->params())
            queue.push(param);

        while (!queue.empty()) {
            Def def = queue.front();
            if (auto primop = def->isa<PrimOp>())
                primops.push_back(primop);
            queue.pop();

            for (auto use : def->uses()) {
                if (use->isa<Lambda>())
                    continue;
                if (use->visit(pass))
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

Schedule schedule_late(const Scope& scope, size_t& pass) {
    Schedule schedule(scope.size());
    Array<std::queue<const PrimOp*>> queues(scope.size());
    pass = scope.world().new_pass();

    for (size_t i = scope.size(); i-- != 0;) {
        auto& queue = queues[i];
        Lambda* cur = scope[i];

        auto fill_queue = [&] (Def def) {
            for (auto op : def->ops()) {
                if (auto primop = op->is_non_const_primop()) {
                    queue.push(primop);

                    if (!primop->visit(pass)) { // init unseen primops
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
            assert(primop->is_visited(pass));

            if (primop->counter == 0) {
                Lambda*& late = get_late(primop);

                if (late == cur) {
                    schedule[late->sid()].push_back(primop);
                    fill_queue(primop);
                } else {
                    late = late ? scope.domtree().lca(cur, late) : cur;
                    queues[late->sid()].push(primop);
                }
            } else
                --primop->counter;
        }
    }

    for (auto& primops : schedule)
        std::reverse(primops.begin(), primops.end());

    return schedule;
}

Schedule schedule_smart(const Scope& scope) {
    Schedule smart(scope.size());
    Schedule early = schedule_early(scope);
    size_t pass;
    schedule_late(scope, pass); // set late pointers in primop and remember pass

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda_early = scope[i];
        for (auto primop : early[i]) {
            if (!primop->is_visited(pass))
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
            smart[lambda_best->sid()].push_back(primop);
        }
    }

    return smart;
}

} // namespace anydsl2
