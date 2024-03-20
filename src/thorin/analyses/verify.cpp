#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"

namespace thorin {

// TODO this needs serious rewriting

static bool verify_calls(World& world, ScopesForest&) {
    bool ok = true;
    for (auto def : world.defs()) {
        if (auto cont = def->isa<Continuation>())
            ok &= cont->verify();
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

#if 0
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
#endif

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
