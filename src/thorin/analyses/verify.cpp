#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

// TODO this needs serious rewriting

static void verify_calls(World& world) {
    for (auto def : world.defs()) {
        if (auto cont = def->isa<Continuation>())
            cont->verify();
    }
}

static void verify_top_level(World& world) {
    Scope::for_each(world, [&] (const Scope& scope) {
        if (scope.has_free_params()) {
            for (auto param : scope.free_params())
                world.ELOG("top-level continuation '{}' got free param '{}' belonging to continuation {}", scope.entry(), param, param->continuation());
            world.ELOG("here: {}", scope.entry());
        }
    });
}

void verify(World& world) {
    verify_calls(world);
    verify_top_level(world);
}

}
