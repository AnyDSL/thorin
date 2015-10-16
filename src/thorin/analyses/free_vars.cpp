#include <vector>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

std::vector<Def> free_vars(const Scope& scope) {
    DefSet vars;
    std::queue<Def> queue;
    DefSet set;

    // now find all params not in scope
    auto enqueue = [&] (Def def) {
        if (!visit(set, def) && !def->is_const()) {
            if (scope._contains(def))
                for (auto op : def->ops())
                    queue.push(op);
            else
                vars.insert(def);
        }
    };

    for (auto lambda : scope) {
        for (auto op : lambda->ops())
            enqueue(op);

        while (!queue.empty())
            enqueue(pop(queue));
    }

    return std::vector<Def>(vars.begin(), vars.end());
}

}
