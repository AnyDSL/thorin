#include <vector>

#include "thorin/memop.h"
#include "thorin/world.h"
#include "thorin/util/queue.h"
#include "thorin/analyses/scope.h"

namespace thorin {

std::vector<Def> free_vars(const Scope& scope) {
    DefSet vars;
    std::queue<Def> queue;
    DefSet set;

    // now find all params not in scope
    auto enqueue = [&] (Def def) {
        if (def->isa<Alloc>() || def->isa<Slot>()) { // HACK;
            vars.insert(def);
        } else {
            for (auto op : def->ops()) {
                if (!visit(set, op)) {
                    if (auto param = op->isa<Param>()) {
                        if (!scope.contains(param->lambda()))
                            vars.insert(param);
                    } else
                        queue.push(op);
                }
            }
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
