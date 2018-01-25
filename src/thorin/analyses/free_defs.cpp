#include "thorin/primop.h"
#include "thorin/analyses/scope.h"

namespace thorin {

DefSet free_defs(const Scope& scope, bool include_closures) {
    DefSet result, done(scope.defs().capacity());
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
            if (!include_closures && primop->isa<Closure>()) {
                result.emplace(primop);
                queue.push(primop->op(1));
                goto queue_next;
            }
            for (auto op : primop->ops()) {
                if ((op->isa<MemOp>() || op->type()->isa<FrameType>()) && !scope.contains(op)) {
                    result.emplace(primop);
                    goto queue_next;
                }
            }

            // HACK for bitcasting address spaces
            if (auto bitcast = primop->isa<Bitcast>()) {
                if (auto dst_ptr = bitcast->type()->isa<PtrType>()) {
                    if (auto src_ptr = bitcast->from()->type()->isa<PtrType>()) {
                        if (       dst_ptr->pointee()->isa<IndefiniteArrayType>()
                                && dst_ptr->addr_space() != src_ptr->addr_space()
                                && !scope.contains(bitcast->from())) {
                            result.emplace(bitcast);
                            goto queue_next;
                        }
                    }
                }
            }

            enqueue_ops(primop);
        } else if (!scope.contains(def))
            result.emplace(def);
queue_next:;
    }

    return result;
}

DefSet free_defs(Continuation* entry) {
    Scope scope(entry);
    return free_defs(scope, true);
}

bool has_free_vars(Continuation* entry) {
    for (auto def : free_defs(entry)) {
        if (!def->isa_continuation())
            return true;
    }

    return false;
}

}
