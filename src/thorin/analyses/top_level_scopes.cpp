#include "thorin/analyses/top_level_scopes.h"

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

AutoVector<Scope*> top_level_scopes(World& world) {
    AutoVector<Scope*> scopes;
    std::queue<Lambda*> queue;
    LambdaSet done;

    auto insert = [&] (Lambda* lambda) { 
        auto scope = new Scope(lambda);
        scopes.emplace_back(scope);

        for (auto lambda : *scope)
            done.insert(lambda); 
    };

    for (auto lambda : world.externals())
        insert(lambda);

    size_t cur = 0;

    while (cur != scopes.size()) {
        auto& scope = *scopes[cur++];

        for (auto lambda : scope) {
            for (auto succ : lambda->succs()) {
                if (!scope.contains(succ))
                    insert(succ);
            }
        }
    }

    return scopes;
}

}
