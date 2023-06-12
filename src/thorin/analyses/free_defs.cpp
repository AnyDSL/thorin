#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/world.h"

namespace thorin {

DefSet spillable_free_defs(ScopesForest& forest, Continuation* entry) {
    DefSet result;
    unique_queue<DefSet> queue;

    //for (auto def: forest.get_scope(entry).free_frontier())
    //    queue.push(def);
    queue.push(entry);
    entry->world().VLOG("Computing free variables for {}", entry);

    while (!queue.empty()) {
        auto free = queue.pop();
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
