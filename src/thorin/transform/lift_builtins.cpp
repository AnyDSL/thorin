#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_vars.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    std::vector<Lambda*> todo;
    Scope::for_each(world, [&] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto lambda = n->lambda();
            if (lambda->is_passed_to_accelerator() && !lambda->is_basicblock())
                todo.push_back(lambda);
        }
    });

    for (auto cur : todo) {
        Scope scope(cur);
        auto vars = free_vars(scope);
#ifndef NDEBUG
        for (auto var : vars)
            assert(var->order() == 0 && "creating a higher-order function");
#endif
        auto lifted = lift(scope, vars);

        std::vector<Use> uses(cur->uses().begin(), cur->uses().end()); // TODO rewrite this
        for (auto use : uses) {
            if (auto ulambda = use->isa_lambda()) {
                if (auto to = ulambda->to()->isa_lambda()) {
                    if (to->is_intrinsic()) {
                        auto oops = ulambda->ops();
                        Array<const Def*> nops(oops.size() + vars.size());
                        std::copy(vars.begin(), vars.end(), std::copy(oops.begin(), oops.end(), nops.begin())); // old ops + former free vars
                        assert(oops[use.index()] == cur);
                        nops[use.index()] = world.global(lifted, lifted->loc(), false, lifted->name);           // update to new lifted lambda
                        ulambda->jump(cur, nops.skip_front(), ulambda->jump_loc());                             // set new args
                        // jump to new top-level dummy function
                        ulambda->update_to(world.lambda(ulambda->arg_fn_type(), to->loc(), to->cc(), to->intrinsic(), to->name));
                    }
                }
            }
        }

        assert(free_vars(Scope(lifted)).empty());
    }
}

}
