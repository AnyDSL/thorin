#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

DefSet free_defs(const Scope& scope) {
    DefSet result, done;
    std::queue<const Def*> queue;

    auto enqueue_ops = [&] (const Def* def) {
        for (auto op : def->ops()) {
            auto p = done.emplace(op);
            if (p.second)
                queue.push(op);
        }
    };

    for (auto continuation : scope)
        enqueue_ops(continuation);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto primop = def->isa<PrimOp>()) {
            for (auto op : primop->ops()) {
                if ((op->isa<MemOp>() || op->isa<Slot>()) && !scope.contains(op)) {
                    result.emplace(primop);
                    goto queue_next;
                }
            }

            if (primop->isa<Bitcast>() && !scope.contains(primop)) // HACK for bitcasting pointer address spaces
                result.emplace(primop);
            else
                enqueue_ops(primop);
        } else if (!scope.contains(def))
            result.emplace(def);
queue_next:;
    }

    return result;
}

}
