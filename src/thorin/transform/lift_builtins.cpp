#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_vars.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    std::vector<Continuation*> todo;
    Scope::for_each(world, [&] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (continuation->is_passed_to_accelerator() && !continuation->is_basicblock())
                todo.push_back(continuation);
        }
    });

    for (auto cur : todo) {
        Scope scope(cur);
        auto vars = free_vars(scope);
#ifndef NDEBUG
        for (auto var : vars)
            assert(var->order() == 0 && "creating a higher-order function");
#endif
        auto lifted = lift(scope, {}, vars);

        std::vector<Use> uses(cur->uses().begin(), cur->uses().end()); // TODO rewrite this
        for (auto use : uses) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (auto callee = ucontinuation->callee()->isa_continuation()) {
                    if (callee->is_intrinsic()) {
                        auto oops = ucontinuation->ops();
                        Array<const Def*> nops(oops.size() + vars.size());
                        std::copy(vars.begin(), vars.end(), std::copy(oops.begin(), oops.end(), nops.begin())); // old ops + former free vars
                        assert(oops[use.index()] == cur);
                        nops[use.index()] = world.global(lifted, lifted->loc(), false, lifted->name);           // update to new lifted continuation
                        ucontinuation->jump(cur, ucontinuation->type_args(), nops.skip_front(), ucontinuation->jump_loc());       // set new args

                        // jump to new top-level dummy function
                        auto ncontinuation = world.continuation(ucontinuation->arg_fn_type(), callee->loc(), callee->cc(), callee->intrinsic(), callee->name);
                        ucontinuation->update_callee(ncontinuation);
                    }
                }
            }
        }

        assert(free_vars(Scope(lifted)).empty());
    }
}

}
