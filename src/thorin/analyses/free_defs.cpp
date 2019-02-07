#include "thorin/primop.h"
#include "thorin/analyses/scope.h"

namespace thorin {

// TODO get rid of this mess
DefSet free_defs(const Scope& scope, bool /*include_closures*/) {
    DefSet result, done(scope.defs().capacity());
    std::queue<const Def*> queue;

    auto enqueue_ops = [&] (const Def* def) {
        for (auto op : def->ops()) {
            auto p = done.emplace(op);
            if (p.second)
                queue.push(op);
        }
    };

    for (auto def : scope.defs()) {
        if (auto lam = def->isa_nominal<Lam>())
            enqueue_ops(lam);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        if (!def->isa_nominal()) {
#if 0
            if (!include_closures && primop->isa<Closure>()) {
                result.emplace(primop);
                queue.push(primop->op(1));
                goto queue_next;
            }
#endif
            for (auto op : def->ops()) {
                if ((op->isa<MemOp>() || op->type()->isa<FrameType>()) && !scope.contains(op)) {
                    result.emplace(def);
                    goto queue_next;
                }
            }

#if 0
            // WTF we should finallly get rid of this
            // HACK for bitcasting address spaces
            if (auto bitcast = def->isa<Bitcast>()) {
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
#endif

            enqueue_ops(def);
        } else if (!scope.contains(def))
            result.emplace(def);
queue_next:;
    }

    return result;
}

DefSet free_defs(Lam* entry) {
    Scope scope(entry);
    return free_defs(scope, true);
}

}
