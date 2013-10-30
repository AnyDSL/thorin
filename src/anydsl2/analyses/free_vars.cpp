#include <queue>
#include <vector>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

std::vector<Def> free_vars(const Scope& scope) {
    std::vector<Def> result;
    std::queue<Def> queue;

    // mark everything within this scope
    const auto pass1 = scope.world().new_pass();

    for (auto lambda : scope.rpo()) {
        lambda->visit_first(pass1);

        for (auto param : lambda->params()) {
            param->visit_first(pass1);
            queue.push(param);
        }

        while (!queue.empty()) {
            auto def = queue.front();
            queue.pop();

            for (auto use : def->uses()) {
                if (!use->isa_lambda() && !use->visit(pass1))
                    queue.push(use);
            }
        }
    }

    // now find everything now marked previously
    const auto pass2 = scope.world().new_pass();

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
