#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"

namespace thorin {

static void verify_calls(World& world) {
    for (auto lambda : world.lambdas()) {
        if (!lambda->empty())
            assert(lambda->to_fn_type()->size() == lambda->arg_fn_type()->size() && "argument/parameter mismatch");
    }
}

void verify(World& world) {
    verify_calls(world);
}

}
