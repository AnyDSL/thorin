#include "anydsl2/analyses/topo_sort.h"

#include <queue>

#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

std::vector<const Def*> topo_sort(const Scope& scope) {
    std::vector<const Def*> result;
    size_t pass = scope.world().new_pass();
    std::queue<const Def*> queue;

    for_all (lambda, scope.rpo()) {
        result.push_back(lambda);

        for_all (param, lambda->params())
            queue.push(param);

        while (!queue.empty()) {
            const Def* def = queue.front();
            result.push_back(def);
            queue.pop();

            for_all (use, def->uses()) {
                if (use->isa<Lambda>())
                    continue;
                if (use->visit(pass))
                    --use->counter;
                else {
                    use->counter = -1;
                    for_all (op, use->ops()) {
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

    return result;
}

}
