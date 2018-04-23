#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    while (true) {
        Continuation* cur = nullptr;
        Scope::for_each(world, [&] (const Scope& scope) {
            if (cur) return;
            for (auto n : scope.f_cfg().post_order()) {
                if (n->continuation() == scope.entry() ||
                    n->continuation()->order() <= 1)
                    continue;
                if (is_passed_to_accelerator(n->continuation())) {
                    cur = n->continuation();
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

        // remove all continuations - they should be top-level functions and can thus be ignored
        std::vector<const Def*> defs;
        for (auto param : free_defs(scope)) {
            if (!param->isa_continuation()) {
                assert(param->order() == 0 && "creating a higher-order function");
                defs.push_back(param);
            }
        }

        auto lifted = lift(scope, defs);
        for (auto use : cur->copy_uses()) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (auto callee = ucontinuation->callee()->isa_continuation()) {
                    if (callee->is_intrinsic()) {
                        auto old_ops = ucontinuation->ops();
                        Array<const Def*> new_ops(old_ops.size() + defs.size());
                        std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin()));    // old ops + former free defs
                        assert(old_ops[use.index()] == cur);
                        new_ops[use.index()] = world.global(lifted, false, lifted->debug());                                // update to new lifted continuation

                        // jump to new top-level dummy function with new args
                        auto fn_type = world.fn_type(Array<const Type*>(new_ops.size()-1, [&] (auto i) { return new_ops[i+1]->type(); }));
                        auto ncontinuation = world.continuation(fn_type, callee->cc(), callee->intrinsic(), callee->debug());
                        ucontinuation->jump(ncontinuation, new_ops.skip_front(), ucontinuation->jump_debug());
                    }
                }
            }
        }

        world.cleanup();
    }
}

}
