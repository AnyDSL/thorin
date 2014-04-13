#include <iostream>

#include "thorin/world.h"
#include "thorin/type.h"
#include "thorin/literal.h"
#include "thorin/memop.h"

namespace thorin {

static void within(World& world, const DefNode* def) {
    assert(world.types().find(*def->type()) != world.types().end());
    if (auto primop = def->isa<PrimOp>()) {
        assert(world.primops().find(primop) != world.primops().end());
    } else if (auto lambda = def->isa_lambda())
        assert(world.lambdas().find(lambda) != world.lambdas().end());
    else
        within(world, def->as<Param>()->lambda());
}

void verify_closedness(World& world) {
    auto check = [&](const DefNode* def) {
        within(world, def->representative_);
        for (auto op : def->ops())
            within(world, op.node());
        for (auto use : def->uses_)
            within(world, use.def().node());
        for (auto r : def->representatives_of_)
            within(world, r);
    };

    for (auto primop : world.primops())
        check(primop);
    for (auto lambda : world.lambdas()) {
        check(lambda);
        for (auto param : lambda->params())
            check(param);
    }
}

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

    while (!queue.empty()) {
        auto def = queue.front();
        queue.pop();
    }
}
#endif

void verify_calls(World& world) {
    for (auto lambda : world.lambdas()) {
        if (!lambda->empty()) {
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
    verify_closedness(world); 
    verify_calls(world);
    //verify_cyclefree(world);
}

//------------------------------------------------------------------------------

} // namespace thorin
