#include <queue>
#include <vector>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

std::vector<Def> free_vars(const Scope& scope) {
    DefSet vars;
    std::queue<Def> queue;
    DefSet set;

    // now find everything not in scope
    auto fill_queue = [&] (Def def) {
        for (auto op : def->ops()) {
            if (op->is_const())
                continue;
            if (!scope.contains(op)) {
                vars.insert(op);
                set.visit(op);
                continue;
            }

            if (!set.visit(op))
                queue.push(op);
        }
    };

    for (auto lambda : scope.rpo()) {
        fill_queue(lambda);

        while (!queue.empty()) {
            auto def = queue.front();
            queue.pop();
            fill_queue(def);
        }
    }

    return std::vector<Def>(vars.begin(), vars.end());
}

}
