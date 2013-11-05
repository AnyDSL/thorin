#include <queue>
#include <vector>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

std::vector<Def> free_vars(const Scope& scope) {
    std::vector<Def> result;
    std::queue<Def> queue;
    const auto pass1 = scope.mark();
    const auto pass2 = scope.world().new_pass();

    // now find everything not marked previously
    auto fill_queue = [&] (Def def) {
        for (auto op : def->ops()) {
            if (op->is_const())
                continue;
            if (op->cur_pass() < pass1) {
                result.push_back(op);
                op->visit_first(pass2);
                continue;
            }

            if (!op->visit(pass2))
                queue.push(op);
        }
    };

    for (auto lambda : scope.rpo()) {
        fill_queue(lambda);

        while (!queue.empty()) {
            auto def = queue.front();
            queue.pop();
            fill_queue(def);
        }
    }

    return result;
}

}
