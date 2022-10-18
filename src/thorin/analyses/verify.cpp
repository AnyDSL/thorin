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

static bool verify_mem(World& world) {
    bool ok = true;
    Scope::for_each(world, [&] (const Scope& scope) {
        for (auto def : free_defs(scope)) {
            if (is_mem(def)) {
                world.ELOG("scope '{}' got free mem '{}' with {} uses", scope.entry(), def, def->num_uses());
                ok = false;
            }
        }
    });
    return ok;
}

void verify(World& world) {
    bool ok = true;
    ok &= verify_calls(world);
    ok &= verify_top_level(world);
    ok &= verify_mem(world);
    if (!ok)
        world.dump();
}

}
