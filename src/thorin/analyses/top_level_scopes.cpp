#include "thorin/analyses/top_level_scopes.h"

#include "thorin/world.h"
#include "thorin/util/queue.h"

namespace thorin {

AutoVector<Scope*> top_level_scopes_deprecated(World& world) {
    AutoVector<Scope*> scopes;
    LambdaSet done;

    auto insert = [&] (Lambda* lambda) {
        if (done.contains(lambda))
            return;
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

void top_level_scopes(World& world, std::function<void(Scope&)> f, bool elide_empty) {
    LambdaSet done;
    std::queue<Lambda*> queue;

    for (auto lambda : world.externals()) {
        assert(!lambda->empty() && "external must not be empty");
        queue.push(lambda);
    }

    while (!queue.empty()) {
        auto lambda = pop(queue);
        if (elide_empty && lambda->empty())
            continue;
        Scope scope(lambda);
        f(scope);
        for (auto lambda : scope)
            done.insert(lambda);

        for (auto lambda : scope) {
            for (auto succ : lambda->succs()) {
                if (!done.contains(succ)) {
                    done.insert(succ);
                    queue.push(succ);
                }
            }
        }
    }
}

}
