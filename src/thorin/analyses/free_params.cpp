#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

DefSet free_params(const Scope& scope) {
    DefSet result;
    DefSet done;
    std::queue<const Def*> queue;

    auto enqueue = [&] (const Def* def) {
        auto p = done.emplace(def);
        if (p.second)
            queue.push(def);
    };

    for (auto continuation : scope)
        enqueue(continuation);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto primop = def->isa<PrimOp>()) {
            for (auto op : primop->ops())
                enqueue(op);
        } else if (!scope.contains(def)) // must be param or primop
            result.emplace(def);
    }

    return result;
}

}
