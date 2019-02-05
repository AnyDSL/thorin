#include "thorin/opt/optimizer.h"

#include "thorin/analyses/scope.h"

namespace thorin {

void Optimizer::run() {
    Scope::for_each(world(), [&](const Scope& scope) {
        scope.post_order_walk(
            [&](Lam* lam) {
                for (auto&& opt : optimizations_) {
                    opt->visit(lam);
                }
            },
            [&](const Def* def) {
                for (auto&& opt : optimizations_) {
                    opt->visit(def);
                }
            }
        );
    });
}

}
