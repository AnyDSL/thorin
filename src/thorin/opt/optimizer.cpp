#include "thorin/opt/optimizer.h"

#include "thorin/analyses/scope.h"
#include "thorin/opt/inliner.h"

namespace thorin {

#if 0
void swap(Optimizer& a, Optimizer& b);
    using std::swap;
    swap(a.world_, b.world_);
    swap(a.opts_,  b.opts_);
}
#endif

void Optimizer::run() {
    for (auto lam : world().externals())
        lams_.push(lam);

    // visits all lambdas
    while (!lams_.empty()) {
        auto lam = lams_.pop();

        for (auto&& opt : opts_)
            opt->visit(lam);

        defs_.push(lam);
        const Def* def = nullptr;

        // post-order walk of all ops within cur
        while (!defs_.empty()) {
            def = defs_.top();

            bool todo = false;
            for (auto op : def->ops()) {
                if (auto lam = op->isa_lam())
                    lams_.push(lam); // queue in outer loop
                else
                    todo |= defs_.push(op);
            }

            if (!todo) {
                for (auto&& opt : opts_) {
                    assert(!def2def_.contains(def));
                    auto new_def = opt->visit(def);
                    def2def_[def] = new_def;
                    def = new_def;
                }

                defs_.pop();
            }
        }

        lam->set_body(def);
    }
}

Optimizer std_optimizer(World& world) {
    Optimizer result(world);
    result.create<Inliner>();
    return result;
}

}
