#include <queue>
#include <vector>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

std::vector<Def> free_vars(const Scope& scope) {
    std::vector<Def> result;
    std::queue<Def> queue;
    const DefSet& pass1 = scope.mark();
    DefSet pass2;

    // now find everything not marked previously
    auto fill_queue = [&] (Def def) {
        for (auto op : def->ops()) {
            if (op->is_const())
                continue;
            if (!pass1.contains(op)) {
                result.push_back(op);
                pass2.visit(op);
                continue;
            }

            if (!pass2.visit(op))
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

    return result;
}

}
