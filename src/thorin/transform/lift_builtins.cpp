#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    std::vector<Continuation*> todo;
    ContinuationSet do_force_inline;
    Scope::for_each(world, [&] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (continuation->is_passed_to_accelerator() && !continuation->is_basicblock()) {
                todo.push_back(continuation);
                if (continuation->is_passed_to_intrinsic(Intrinsic::Vectorize))
                    do_force_inline.emplace(continuation);
            }
        }
    });

    static const int inline_threshold = 4;
    for (auto continuation : do_force_inline) {
        Scope scope(continuation);
        force_inline(scope, inline_threshold);
    }

    for (auto cur : todo) {
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

        std::vector<Use> uses(cur->uses().begin(), cur->uses().end()); // TODO rewrite this
        for (auto use : uses) {
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
    }
}

}
