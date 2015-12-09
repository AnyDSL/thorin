#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

void lower2cff(World& world) {
    HashMap<Call, Lambda*> cache;
    LambdaSet top;

    bool local = true;
    for (bool todo = true; todo || local;) {
        todo = false;

        Scope::for_each(world, [&] (Scope& scope) {
            bool dirty = false;

            auto is_bad = [&] (Lambda* to) {
                if (to->empty())
                    return false;
                if (local)
                    return scope.inner_contains(to) && !to->is_basicblock();
                else {
                    if (top.contains(to))
                        return !to->is_returning() && !scope.outer_contains(to);
                    else
                        return !to->is_basicblock();
                }
            };

            const auto& cfg = scope.f_cfg();
            for (auto n : cfg.post_order()) {
                auto lambda = n->lambda();
                if (auto to = lambda->to()->isa_lambda()) {
                    if (is_bad(to)) {
                        DLOG("bad: %", to);
                        todo = dirty = true;

                        Array<Def> ops(lambda->size());
                        ops.front() = to;
                        for (size_t i = 1, e = ops.size(); i != e; ++i)
                            ops[i] = to->param(i-1)->order() > 0 ? lambda->arg(i-1) : nullptr;


                        Array<Type> type_args(lambda->num_type_args());
                        Call call(std::move(type_args), std::move(ops));

                        const auto& p = cache.emplace(call, nullptr);
                        Lambda*& target = p.first->second;
                        if (p.second) {
                            target = drop(call); // use already dropped version as target
                        }

                        // TODO type_args!!!
                        //jump_to_cached_call(lambda, target, call);
                    }
                }
            }

            if (dirty)
                scope.update();
            top.insert(scope.entry());
        });

        if (!todo && local) {
            DLOG("switching to global mode");
            local = false;
            todo = true;
        }
    }

    debug_verify(world);
    world.cleanup();
}

}
