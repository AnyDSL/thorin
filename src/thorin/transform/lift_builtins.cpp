#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    while (true) {
        Lam* cur = nullptr;
        Scope::for_each(world, [&] (const Scope& scope) {
            if (cur) return;
            for (auto n : scope.f_cfg().post_order()) {
                if (n->lam()->type()->order() <= 1)
                    continue;
                if (is_passed_to_accelerator(n->lam(), false)) {
                    cur = n->lam();
                    break;
                }
            }
        });

        if (!cur) break;

        static const int inline_threshold = 4;
        if (is_passed_to_intrinsic(cur, Intrinsic::Vectorize)) {
            Scope scope(cur);
            force_inline(scope, inline_threshold);
        }

        Scope scope(cur);

        // remove all lams - they should be top-level functions and can thus be ignored
        std::vector<const Def*> defs;
        for (auto param : free_defs(scope)) {
            if (!param->isa_lam()) {
                assert(param->type()->order() == 0 && "creating a higher-order function");
                defs.push_back(param);
            }
        }

        auto lifted = lift(scope, defs);
        for (auto use : cur->copy_uses()) {
            if (auto ulam = use->isa_lam()) {
                if (auto callee = ulam->app()->callee()->isa_lam()) {
                    if (callee->is_intrinsic()) {
                        auto old_ops = ulam->ops();
                        Array<const Def*> new_ops(old_ops.size() + defs.size());
                        std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin()));    // old ops + former free defs
                        assert(old_ops[use.index()] == cur);
                        new_ops[use.index()] = world.global(lifted, false, lifted->debug());                                // update to new lifted lam

                        // jump to new top-level dummy function with new args
                        auto cn = world.cn(Array<const Def*>(new_ops.size()-1, [&] (auto i) { return new_ops[i+1]->type(); }));
                        auto nlam = world.lam(cn, callee->cc(), callee->intrinsic(), callee->debug());
                        ulam->app(nlam, new_ops.skip_front(), ulam->app()->debug());
                    }
                }
            }
        }

        world.cleanup();
    }
}

}
