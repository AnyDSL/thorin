#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

void partial_evaluation(World& world) {
    bool todo = true;

    while (todo) {
        todo = false;
        LambdaSet lambdas = world.lambdas();
        for (auto lambda : lambdas) {
            if (auto to = lambda->to()->isa_lambda()) {
                if (lambda->attr().is_run()) {
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
                        lambda->jump(drop(Scope(to), indices, with, generic_map), args);
                        todo = true;
                    }

                    lambda->attr().unset_run();
                    Scope scope(lambda);
                    auto& looptree = scope.looptree();
                    for (auto loop_lambda : looptree.root()->child(0)->headers())
                        std::cout << loop_lambda->unique_name() << std::endl;
                }
            }
        }

        merge_lambdas(world);
        world.cleanup();
    }

    world.opt();
}

}
