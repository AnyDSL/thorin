#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

static void verify_calls(World& world) {
    for (auto continuation : world.continuations()) {
        if (!continuation->empty())
            assert(continuation->callee_fn_type()->num_ops() == continuation->arg_fn_type()->num_ops() && "argument/parameter mismatch");
    }
}

enum Color {
    White, Gray, Black
};

static void visit(PrimOpMap<Color>& primop2color, const PrimOp* primop) {
    auto& color = primop2color.find(primop)->second;
    if (color == White) {
        color = Gray;
        for (auto op : primop->ops()) {
            if (auto primop = op->isa<PrimOp>())
                visit(primop2color, primop);
        }
        color = Black;
    } else if (color == Gray)
        ELOG("detected primop cycle at %:", primop);
}

static void verify_primops(World& world) {
    PrimOpMap<Color> primop2color;

    for (auto primop : world.primops())
        primop2color[primop] = White;

    for (auto primop : world.primops())
        visit(primop2color, primop);
}

void verify(World& world) {
    verify_calls(world);
    verify_primops(world);
}

}
