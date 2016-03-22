#include <vector>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

std::vector<const Def*> free_vars(const Scope& scope) {
    DefSet vars;
    std::queue<const Def*> queue;
    DefSet set;

    // now find all params not in scope
    auto enqueue = [&] (const Def* def) {
        if (!visit(set, def) && !def->is_const()) {
            if (scope.contains(def))
                for (auto op : def->ops())
                    queue.push(op);
            else
                vars.insert(def);
        }
    };

    for (auto continuation : scope) {
        for (auto op : continuation->ops())
            enqueue(op);

        while (!queue.empty())
            enqueue(pop(queue));
    }

    return std::vector<const Def*>(vars.begin(), vars.end());
}

}
