#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"

namespace thorin {

// TODO this needs serious rewriting

static bool verify_calls(World& world, ScopesForest& forest) {
    bool ok = true;
    for (auto def : world.defs()) {
        if (auto cont = def->isa<Continuation>())
            ok &= cont->verify();
        if (auto closure = def->isa<Closure>()) {
            Scope& s = forest.get_scope(closure->fn());
            if (s.parent_scope()) {
                world.ELOG("closure '{}' has a non-top level function: {}", def, closure->fn());
                ok = false;
            }

            unique_queue<DefSet> env;
            env.push(closure->env());

            while (!env.empty()) {
                auto e = env.pop();
                if (auto tuple = e->isa<Tuple>()) {
                    for (auto o : tuple->ops())
                        env.push(o);
                } else if (auto heap = e->isa<Cell>()) {
                    env.push(heap->contents());
                } else if (auto cont = e->isa_nom<Continuation>()) {
                    world.ELOG("closure '{}' has a continuation in it's environment {}", def, cont);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

static bool verify_top_level(World& world, ScopesForest& forest) {
    bool ok = true;
    unique_queue<DefSet> defs;
    for (auto& external : world.externals())
        defs.push(external.second);
    while (!defs.empty()) {
        auto def = defs.pop();
        if (auto cont = def->isa_nom<Continuation>()) {
            world.VLOG("verifying external continuation '{}'", cont);
            auto& scope = forest.get_scope(cont);
            scope.verify();
            if (scope.has_free_params()) {
                for (auto param : scope.free_params())
                    world.ELOG("top-level continuation '{}' got free param '{}' belonging to continuation {}", scope.entry(), param, param->continuation());
                world.ELOG("here: {}", scope.entry());
                ok = false;
            }
        } else {
            for (auto op : def->ops())
                defs.push(op);
        }
    }
    for (auto cont : world.copy_continuations()) {
        auto& scope = forest.get_scope(cont);
        scope.verify();
    }
    return ok;
}

static bool verify_param(World& world) {
    bool ok = true;
    for (auto def : world.defs()) {
        if (auto param = def->isa<Param>()) {
            auto cont = param->continuation();
            if (cont->dead_) {
                world.ELOG("param '{}' originates in dead continuation {}", param, cont);
                ok = false;
            }
        }
    }
    return ok;
}

void verify(World& world) {
    ScopesForest forest(world);
    bool ok = true;
    ok &= verify_calls(world, forest);
    ok &= verify_top_level(world, forest);
    //TODO: This should not fail!
    //ok &= verify_param(world);
    if (!ok)
        world.dump();
    assert(ok);
}

}
