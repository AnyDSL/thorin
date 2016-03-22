#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"

namespace thorin {

static void verify_calls(World& world) {
    for (auto continuation : world.continuations()) {
        if (!continuation->empty())
            assert(continuation->callee_fn_type()->size() == continuation->arg_fn_type()->size() && "argument/parameter mismatch");
    }
}

void verify(World& world) {
    verify_calls(world);
}

}
