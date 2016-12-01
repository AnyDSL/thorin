#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

void lower2cff(World& world) {
    HashMap<Call, Continuation*> cache;
    ContinuationSet top;

    bool local = true;
    for (bool todo = true; todo || local;) {
        todo = false;

        Scope::for_each(world, [&] (Scope& scope) {
            bool dirty = false;

            auto is_bad = [&] (Continuation* to) {
                if (to->empty())
                    return false;
                if (local)
                    return scope.inner_contains(to) && !to->is_basicblock();
                else {
                    if (top.contains(to))
                        return !to->is_returning() && !scope.contains(to);
                    else
                        return !to->is_basicblock();
                }
            };

            const auto& cfg = scope.f_cfg();
            for (auto n : cfg.post_order()) {
                auto continuation = n->continuation();
                if (auto callee = continuation->callee()->isa_continuation()) {
                    if (is_bad(callee)) {
                        DLOG("bad: %", callee);
                        todo = dirty = true;

                        Call call(continuation);
                        call.callee() = callee;
                        for (size_t i = 0, e = call.num_args(); i != e; ++i)
                            call.arg(i) = callee->param(i)->order() > 0 ? continuation->arg(i) : nullptr;

                        const auto& p = cache.emplace(call, nullptr);
                        Continuation*& target = p.first->second;
                        if (p.second)
                            target = drop(call); // use already dropped version as target

                        jump_to_cached_call(continuation, target, call);
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
