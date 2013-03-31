#include "anydsl2/transform/partial_evaluation.h"

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

void partial_evaluation(World& world) {
    bool todo = true;

    while (todo) {
        todo = false;
        LambdaSet lambdas = world.lambdas();
        for_all (lambda, lambdas) {
            if (Lambda* to = lambda->to()->isa_lambda()) {
                size_t num_args = lambda->num_args();
                Array<size_t> indices(num_args);
                Array<const Def*> with(num_args);
                size_t x = 0;
                std::vector<const Def*> args;
                for (size_t i = 0; i != num_args; ++i) {
                    const Def* arg = lambda->arg(i);
                    if (arg->is_const()) {
                        indices[x] = i;
                        with[x++] = arg;
                    } else
                        args.push_back(arg);
                }

                if (x) {
                    indices.shrink(x);
                    with.shrink(x);
                    GenericMap generic_map;
                    bool res = to->type()->infer_with(generic_map, lambda->arg_pi());
                    assert(res);
                    lambda->jump(Scope(to).drop(indices, with, generic_map), args);
                    todo = true;
                }
            }
        }

        //std::cout << world.lambdas().size() << std::endl;

        merge_lambdas(world);
        world.cleanup();
    }
}

}
