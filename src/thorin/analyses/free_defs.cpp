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

DefSet spillable_free_defs(ScopesForest& forest, Continuation* entry) {
    DefSet result;
    unique_queue<DefSet> queue;

    //for (auto def: forest.get_scope(entry).free_frontier())
    //    queue.push(def);
    queue.push(entry);

    while (!queue.empty()) {
        auto free = queue.pop();
        // assert(!free->type()->isa<ReturnType>());
        assert(!free->type()->isa<MemType>());

        //if (free == entry)
        //    continue;

        if (auto cont = free->isa_nom<Continuation>()) {
            auto& scope = forest.get_scope(cont);
            auto& frontier = scope.free_frontier();
            entry->world().VLOG("encountered: {}, frontier_size={}", cont, frontier.size());
            for (auto def: frontier) {
                queue.push(def);
            }
            continue;
        }

        if (free->has_dep(Dep::Param)) {
            entry->world().VLOG("fv: {} : {}", free, free->type());
            result.insert(free);
        } else
            free->world().WLOG("ignoring {} because it has no Param dependency", free);
    }

    return result;
}

}
