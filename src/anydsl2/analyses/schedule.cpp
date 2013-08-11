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
    std::queue<const Def*> queue;
    const size_t pass = scope.world().new_pass();

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda = scope[i];
        std::vector<const PrimOp*>& primops = schedule[i];

        for (auto param : lambda->params())
            queue.push(param);

        while (!queue.empty()) {
            const Def* def = queue.front();
            if (const PrimOp* primop = def->isa<PrimOp>())
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
    std::queue<const Def*> queue;
    pass = scope.world().new_pass();

    for (size_t i = scope.size(); i-- != 0;) {
        Lambda* cur = scope[i];
        queue.push(cur);

        while (!queue.empty()) {
            const Def* def = queue.front();
            queue.pop();

            for (auto op : def->ops()) {
                if (const PrimOp* primop = op->is_non_const_primop()) {
                    if (!primop->visit(pass)) {     // init unseen primops
                        primop->ptr = 0;
                        primop->counter = primop->num_uses();
                    }

                    Lambda*& late = get_late(primop);
                    late = late ? scope.domtree().lca(cur, late) : cur;

                    if (--primop->counter == 0) {   // only visit once when counter == 0
                        schedule[late->sid()].push_back(primop);
                        queue.push(primop);
                    }

                    assert(primop->counter != size_t(-1));
                }
            }
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
                continue;                       // primop is dead
            Lambda* lambda_best = get_late(primop);
            assert(scope.contains(lambda_best));
            if (primop->isa<Slot>() || primop->isa<Enter>())
                lambda_best = lambda_early;     // place these guys always early
            else if (!primop->isa<Leave>()) {   // place this guy always late
                // all other guys are placed as late as possible but keep them out of loops, please
                int depth = std::numeric_limits<int>::max();
                for (Lambda* i = lambda_best; i != lambda_early; i = scope.domtree().idom(i)) {
                    int cur_depth = scope.looptree().depth(i);
                    if (cur_depth < depth) {
                        lambda_best = i;
                        depth = cur_depth;
                    }
                }
            }
            smart[lambda_best->sid()].push_back(primop);
        }
    }

    return smart;
}

} // namespace anydsl2
