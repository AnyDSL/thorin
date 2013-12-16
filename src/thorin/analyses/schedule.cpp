#include <algorithm>
#include <iostream>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"

namespace thorin {

typedef LambdaMap<std::vector<const PrimOp*>> Schedule;

static bool sort_primops(Def def1, Def def2) {
    if (def1->kind() == def2->kind())
        return def1->gid() < def2->gid();

    if ( def1->isa<Enter>() || def1->isa<Slot>() || def1->isa<Load>() || def1->isa<Store>())
        return true;

    return def1->gid() < def2->gid();
}

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

            std::vector<Def> todo;
            for (auto use : def->uses()) {
                if (use->isa<Lambda>())
                    continue;
                if (set.visit(use))
                    --num_placed[use];
                else {
                    num_placed[use] = -1;
                    for (auto op : use->ops()) {
                        if (scope.contains(op))
                            ++num_placed[use];
                    }
                }
                assert(num_placed[use] != size_t(-1));

                if (num_placed[use] == 0)
                    todo.push_back(use);
            }

            std::sort(todo.begin(), todo.end(), sort_primops);
            for (auto def : todo)
                queue.push(def);
        }
    }

    return schedule;
}

static Schedule schedule_late(const Scope& scope, DefMap<Lambda*> &def2late) {
    DefMap<int> def2num;
    std::vector<Def> zero;

    for (auto def : scope.in_scope()) {
        if (auto primop = def->isa<PrimOp>()) {
            int num = 0;
            for (auto use : primop->uses()) {
                if (scope.contains(use))
                    ++num;
            }
            if (num != 0) // not dead
                def2num[def] = num;
        }
    }

    Schedule schedule;
    const DomTree domtree(scope);
    assert(def2late.empty());

    for (Lambda* cur : scope.backwards_rpo()) {
        auto decrease = [&] (Def def) {
            assert(scope.contains(def));
            for (auto op : def->ops()) {
                if (op->isa<PrimOp>() && scope.contains(op)) {
                    Lambda*& late = def2late[op];
                    late = late == nullptr ? cur : domtree.lca(late, cur);

                    assert(def2num.find(op) != def2num.end());
                    if (--def2num[op] == 0)
                        zero.push_back(op);
                    assert(def2num[op] >= 0);
                }
            }
        };

        decrease(cur);
        def2late[cur] = cur;

        bool todo = true;
        do {
            std::vector<const PrimOp*> remove;
            std::sort(zero.begin(), zero.end(), [] (Def def1, Def def2) { return !sort_primops(def1, def2); });

            for (auto z : zero) {
                const PrimOp* primop = z->as<PrimOp>();
                auto late = def2late[primop];
                assert(late);
                schedule[late].push_back(primop);
                remove.push_back(primop);
            }

            if (zero.empty())
                todo = false;
            else
                zero.clear();

            for (auto op : remove)
                decrease(op);
        } while (todo);
    }
    assert(zero.empty());

    for (auto& primops : schedule)
        std::reverse(primops.second.begin(), primops.second.end());

    return schedule;
}

Schedule schedule_late(const Scope& scope) { DefMap<Lambda*> late; return schedule_late(scope, late); }

Schedule schedule_smart(const Scope& scope) {
    Schedule smart;
    const DomTree domtree(scope); // TODO cache domtree across schedule_late
    const LoopTree looptree(scope);
    Schedule early = schedule_early(scope);
    DefMap<Lambda*> def2late;
    schedule_late(scope, def2late); // set late pointers in primop and remember pass

    for (size_t i = 0, e = scope.size(); i != e; ++i) {
        Lambda* lambda_early = scope[i];
        for (auto primop : early[lambda_early]) {
            if (!def2late.contains(primop))
                continue;       // primop is dead
            Lambda* lambda_best = def2late[primop];
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
