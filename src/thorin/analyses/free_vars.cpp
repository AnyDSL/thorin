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
        if (!visit(set, def)) {
            if (auto param = def->isa<Param>()) {
                if (!scope.contains(param->lambda()))
                    vars.insert(param);
                }
            else if (auto primop = def->isa<PrimOp>()) {
                if (primop->isa<Alloc>() || primop->isa<Slot>()) { // HACK;
                    vars.insert(primop);
                } else {
                    for (auto op : primop->ops())
                        queue.push(op);
                }
            }
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
