#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void force_inline(Scope& scope, int threshold) {
    for (bool todo = true; todo && threshold-- != 0;) {
        todo = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (!continuation->has_body()) continue;
            if (auto callee = continuation->body()->callee()->isa_continuation()) {
                if (callee->has_body() && !scope.contains(callee)) {
                    Scope callee_scope(callee);
                    continuation->jump(drop(callee_scope, continuation->body()->args()), {}, continuation->debug()); // TODO debug
                    todo = true;
                }
            }
        }

        if (todo)
            scope.update();
    }

    for (auto n : scope.f_cfg().reverse_post_order()) {
        auto continuation = n->continuation();
        if (!continuation->has_body()) continue;
        if (auto callee = continuation->body()->callee()->isa_continuation()) {
            if (callee->has_body() && !scope.contains(callee))
                scope.world().WLOG("couldn't inline {} at {} within scope of {}", callee, continuation->loc(), scope.entry());
        }
    }
}

void inliner(World& world) {
    world.VLOG("start inliner");

    static const int factor = 4;
    static const int offset = 4;

    ContinuationMap<std::unique_ptr<Scope>> continuation2scope;

    auto get_scope = [&] (Continuation* continuation) -> Scope* {
        auto i = continuation2scope.find(continuation);
        if (i == continuation2scope.end())
            i = continuation2scope.emplace(continuation, std::make_unique<Scope>(continuation)).first;
        return i->second.get();
    };

    auto is_candidate = [&] (Continuation* continuation) -> Scope* {
        if (continuation->has_body() && continuation->order() > 1) {
            auto scope = get_scope(continuation);
            if (scope->defs().size() < scope->entry()->num_params() * factor + offset) {
                // check that the function is not recursive to prevent inliner from peeling loops
                for (auto& use : continuation->uses()) {
                    // note that if there was an edge from parameter to continuation,
                    // we would need to check if the use is a parameter here.
                    if (!use->isa<Param>() && scope->contains(use.def()))
                        return nullptr;
                }
                return scope;
            }
        }
        return nullptr;
    };

    Scope::for_each(world, [&] (Scope& scope) {
        bool dirty = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (!continuation->has_body()) continue;
            if (auto callee = continuation->body()->callee()->isa_continuation()) {
                if (callee == scope.entry())
                    continue; // don't inline recursive calls
                world.DLOG("callee: {}", callee);
                if (auto callee_scope = is_candidate(callee)) {
                    world.DLOG("- here: {}", continuation);
                    continuation->jump(drop(*callee_scope, continuation->body()->args()), {}, continuation->debug()); // TODO debug
                    dirty = true;
                }
            }
        }

        if (dirty) {
            scope.update();

            if (auto s = get_scope(scope.entry()))
                s->update();
        }
    });

    world.VLOG("stop inliner");
    debug_verify(world);
    world.cleanup();
}

}
