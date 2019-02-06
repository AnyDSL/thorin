#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

void force_inline(Scope& scope, int threshold) {
    for (bool todo = true; todo && threshold-- != 0;) {
        todo = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto lam = n->lam();
            if (auto app = lam->app()) {
                if (auto callee = app->callee()->isa_nominal<Lam>()) {
                    if (!callee->is_empty() && !scope.contains(callee)) {
                        Scope callee_scope(callee);
                        lam->app(drop(callee_scope, app->args()), Defs{}, app->debug());
                        todo = true;
                    }
                }
            }
        }

        if (todo)
            scope.update();
    }

    for (auto n : scope.f_cfg().reverse_post_order()) {
        auto lam = n->lam();
        if (auto app = lam->app()) {
            if (auto callee = app->callee()->isa_nominal<Lam>()) {
                if (!callee->is_empty() && !scope.contains(callee))
                    WLOG("couldn't inline {} at {} within scope of {}", callee, app->loc(), scope.entry());
            }
        }
    }
}

void inliner(World& world) {
    VLOG("start inliner");

    static const int factor = 4;
    static const int offset = 4;

    LamMap<std::unique_ptr<Scope>> lam2scope;

    auto get_scope = [&] (Lam* lam) -> Scope* {
        auto i = lam2scope.find(lam);
        if (i == lam2scope.end())
            i = lam2scope.emplace(lam, std::make_unique<Scope>(lam)).first;
        return i->second.get();
    };

    auto is_candidate = [&] (Lam* lam) -> Scope* {
        if (!lam->is_empty() && lam->type()->order() > 1) {
            auto scope = get_scope(lam);
            if (scope->defs().size() < scope->entry()->num_params() * factor + offset) {
                // check that the function is not recursive to prevent inliner from peeling loops
                for (auto& use : lam->uses()) {
                    // note that if there was an edge from parameter to lam,
                    // we would need to check if the use is a parameter here.
                    if (scope->contains(use.def()))
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
            auto lam = n->lam();
            if (auto app = lam->app()) {
                if (auto callee = app->callee()->isa_nominal<Lam>()) {
                    if (callee == scope.entry())
                        continue; // don't inline recursive calls
                    DLOG("callee: {}", callee);
                    if (auto callee_scope = is_candidate(callee)) {
                        DLOG("- here: {}", lam);
                        lam->app(drop(*callee_scope, app->args()), Defs{}, app->debug());
                        dirty = true;
                    }
                }
            }
        }

        if (dirty) {
            scope.update();

            if (auto s = get_scope(scope.entry()))
                s->update();
        }
    });

    VLOG("stop inliner");
    debug_verify(world);
    world.cleanup();
}

}
