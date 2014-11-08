#include <iostream>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"

namespace thorin {

#if 0
void verify_cyclefree(World& world) {
    DefSet done;
    PrimOpSet primops;
    LambdaSet lambdas;
    std::queue<Def> queue;

    for (auto cur : world.lambdas()) {
        for (auto op : cur->ops()) {
            if (!done.contains(op)) {
                if (auto primop = op->isa<PrimOp>()) {
                    queue.push(primop);
                } else if (auto lambda = op->isa_lambda()) {
                } else {
                    auto param = op->as<Param>();
                }
            }
        }
    }

    while (!queue.empty())
        pop(queue.front());
}
#endif

void verify_calls(World& world) {
    for (auto lambda : world.lambdas()) {
        if (!lambda->empty()) {
            assert(lambda->to_fn_type()->num_args() == lambda->arg_fn_type()->num_args() && "argument/parameter mismatch");
            // TODO
#if 0
            if (!lambda->to_pi()->check_with(lambda->arg_pi())) {
                std::cerr << "call in '" << lambda->unique_name() << "' broken" << std::endl;
                lambda->dump_jump();
                std::cerr << "to type:" << std::endl;
                lambda->to_pi()->dump();
                std::cerr << "argument type:" << std::endl;
                lambda->arg_pi()->dump();
                std::abort();
            }
            GenericMap map;
            auto res = lambda->to_pi()->infer_with(map, lambda->arg_pi());
            assert(res);
            assert(lambda->to_pi()->specialize(map) == lambda->arg_pi()->specialize(map));
#endif
        }
    }
}

//------------------------------------------------------------------------------

void verify(World& world) {
    verify_calls(world);
    //verify_cyclefree(world);
}

//------------------------------------------------------------------------------

} // namespace thorin
