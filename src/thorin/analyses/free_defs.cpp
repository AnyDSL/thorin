#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/world.h"

namespace thorin {

DefSet spillable_free_defs(const Scope& scope) {
    DefSet result;
    unique_queue<DefSet> queue;

    for (auto def: scope.free_frontier())
        queue.push(def);

    while (!queue.empty()) {
        auto free_def = queue.pop();
        assert(!scope.contains(free_def));
        auto cont = free_def->isa_nom<Continuation>();
        if (cont && cont->is_intrinsic()) {
            scope.world().WLOG("ignoring {} because it is an intrinsic", cont);
            continue;
        }

        assert(!free_def->type()->isa<ReturnType>());
        assert(!free_def->type()->isa<MemType>());

        if (cont == scope.entry())
            continue;

        if (free_def->has_dep(Dep::Param) || cont) {
            scope.world().VLOG("fv: {} : {}", free_def, free_def->type());
            result.insert(free_def);
        } else
            scope.world().WLOG("ignoring {} because it has no Param dependency {}", free_def, cont == nullptr);

        /*
        // HACK for bitcasting address spaces
        if (auto bitcast = free_def->isa<Bitcast>()) {
            if (auto dst_ptr = bitcast->type()->isa<PtrType>()) {
                if (auto src_ptr = bitcast->from()->type()->isa<PtrType>()) {
                    if (dst_ptr->pointee()->isa<IndefiniteArrayType>() && dst_ptr->addr_space() != src_ptr->addr_space() && !scope.contains(bitcast->from())) {
                        result.emplace(bitcast);
                        continue;
                    }
                }
            }
        }*/
    }

    return result;
}

}
