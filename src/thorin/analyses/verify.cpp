#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/free_defs.h"

namespace thorin {

// TODO this needs serious rewriting

static bool verify_calls(World& world) {
    bool ok = true;
    for (auto def : world.defs()) {
        if (auto cont = def->isa<Continuation>())
            ok &= cont->verify();
    }
    return ok;
}

static bool verify_top_level(World& world) {
    bool ok = true;
    Scope::for_each(world, [&] (const Scope& scope) {
        if (scope.has_free_params()) {
            for (auto param : scope.free_params())
                world.ELOG("top-level continuation '{}' got free param '{}' belonging to continuation {}", scope.entry(), param, param->continuation());
            world.ELOG("here: {}", scope.entry());
            ok = false;
        }
    });
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
    bool ok = true;
    ok &= verify_calls(world);
    ok &= verify_top_level(world);
    //TODO: This should not fail!
    //ok &= verify_param(world);
    if (!ok)
        world.dump();
    assert(ok);
}

}
