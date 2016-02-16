#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

enum class EvalState {
    Run, Hlt, None
};

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
                auto to = lambda->to();
                Def end;
                EvalState state = EvalState::None;
                Location loc;

                if (auto evalop = to->isa<EvalOp>()) {
                    to = evalop->begin();
                    end = evalop->end();
                    state = evalop->isa<Run>() ? EvalState::Run : EvalState::Hlt;
                    loc = evalop->loc();
                }

                if (auto to_lambda = to->isa_lambda()) {
                    if (is_bad(to_lambda)) {
                        DLOG("bad: %", to_lambda);
                        todo = dirty = true;

                        Call call(lambda);
                        for (size_t i = 0, e = call.num_type_args(); i != e; ++i)
                            call.type_arg(i) = lambda->type_arg(i);

                        call.to() = to_lambda;
                        for (size_t i = 0, e = call.num_args(); i != e; ++i)
                            call.arg(i) = to_lambda->param(i)->order() > 0 ? lambda->arg(i) : nullptr;


                        const auto& p = cache.emplace(call, nullptr);
                        Lambda*& target = p.first->second;
                        if (p.second) {
                            target = drop(call); // use already dropped version as target
                        }

                        jump_to_cached_call(lambda, target, call);
                        switch (state) {
                            case EvalState::Run: lambda->update_to(world.run(lambda->to(), end, loc));
                            case EvalState::Hlt: lambda->update_to(world.hlt(lambda->to(), end, loc));
                            default: break;
                        }
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
