#include "thorin/world.h"
#include "thorin/analyses/free_vars.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void lift_builtins(World& world) {
    std::vector<Lambda*> todo;
    for (auto cur : world.copy_lambdas()) {
        if (cur->is_passed_to_accelerator() && !cur->is_basicblock())
            todo.push_back(cur);
    }

    for (auto cur : todo) {
        Scope scope(cur);
        auto vars = free_vars(scope);
#ifndef NDEBUG
        for (auto var : vars)
            assert(var->order() == 0 && "creating a higher-order function");
#endif
        auto lifted = lift(scope, vars);

        for (auto use : cur->uses()) {
            if (auto ulambda = use->isa_lambda()) {
                if (auto to = ulambda->to()->isa_lambda()) {
                    if (to->is_intrinsic()) {
                        auto oops = ulambda->ops();
                        Array<Def> nops(oops.size() + vars.size());
                        std::copy(oops.begin(), oops.end(), nops.begin());               // copy over old ops
                        assert(oops[use.index()] == cur);
                        nops[use.index()] = world.global(lifted, false, lifted->name);   // update to new lifted lambda
                        std::copy(vars.begin(), vars.end(), nops.begin() + oops.size()); // append former free vars
                        ulambda->jump(cur, nops.slice_from_begin(1));                    // set new args
                        // jump to new top-level dummy function
                        ulambda->update_to(world.lambda(ulambda->arg_fn_type(), to->cc(), to->intrinsic(), to->name));
                    }
                }
            }
        }

        assert(free_vars(Scope(lifted)).empty());
    }
}

}
