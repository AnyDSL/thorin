#include "thorin/analyses/top_level_scopes.h"

#include "thorin/world.h"
#include "thorin/util/queue.h"

namespace thorin {

template<bool elide_empty>
void top_level_scopes(World& world, std::function<void(Scope&)> f) {
    LambdaSet done;
    std::queue<Lambda*> queue;

    for (auto lambda : world.externals()) {
        assert(!lambda->empty() && "external must not be empty");
        done.insert(lambda);
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

template void top_level_scopes<true> (World&, std::function<void(Scope&)>);
template void top_level_scopes<false>(World&, std::function<void(Scope&)>);

}
