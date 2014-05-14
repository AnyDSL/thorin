#include <vector>

#include "thorin/world.h"
#include "thorin/util/queue.h"
#include "thorin/analyses/scope.h"

namespace thorin {

std::vector<Def> free_vars(const Scope& scope) {
    DefSet vars;
    std::queue<Def> queue;
    DefSet set;

    // now find everything not in scope
    auto enqueue = [&] (Def def) {
        for (auto op : def->ops()) {
            if (op->is_const())
                continue;
            if (!scope.contains(op)) {
                vars.insert(op);
                visit(set, op);
                continue;
            }

            if (!visit(set, op))
                queue.push(op);
        }
    };

    for (auto lambda : scope.rpo()) {
        enqueue(lambda);

        while (!queue.empty())
            enqueue(pop(queue));
    }

    return std::vector<Def>(vars.begin(), vars.end());
}

}
