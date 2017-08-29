#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/log.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(Scope& scope)
        : scope_(scope)
    {}
    ~PartialEvaluator() {
        scope(); // trigger update if dirty
    }

    World& world() { return scope_.world(); }
    Scope& scope() {
        if (dirty_) {
            scope_.update();
            dirty_ = false;
        }
        return scope_;
    }

    void run();

private:
    Scope& scope_;
    HashMap<Call, Continuation*> cache_;
    bool dirty_ = false;
};

void PartialEvaluator::run() {
    // TODO
}

void eat_pe_info(Continuation* cur, bool eval) {
    World& world = cur->world();
    assert(cur->arg(1)->type() == world.ptr_type(world.indefinite_array_type(world.type_pu8())));
    auto msg = cur->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();
    ILOG(cur->callee(), "{}pe_info: {}: {}", eval ? "" : "NOT evaluated: ", msg->as_string(), cur->arg(2));
    auto next = cur->arg(3);
    cur->jump(next, {cur->arg(0), world.tuple({})}, cur->jump_debug());
}

void eat_pe_known(Continuation* cur, bool eval) {
    World& world = cur->world();
    auto val = cur->arg(1);
    auto next = cur->arg(2);
    cur->jump(next, {cur->arg(0), world.literal(eval && is_const(val))}, cur->jump_debug());
}

bool eat_intrinsic(Intrinsic intrinsic, Continuation* cur, bool eval) {
    switch (intrinsic) {
        case Intrinsic::PeInfo:  eat_pe_info (cur, eval); return true;
        case Intrinsic::PeKnown: eat_pe_known(cur, eval); return true;
        default: return false;
    }
}

//------------------------------------------------------------------------------

template <typename F>
void eval_intrinsics(World& world, F f) {
    Scope::for_each(world, [&] (Scope& scope) {
        bool dirty = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa<Continuation>()) {
                dirty |= f(callee, continuation);
            }
        }

        if (dirty)
            scope.update();
    });
}

void eval(World& world) {
    Scope::for_each(world, [&] (Scope& scope) {
        PartialEvaluator partial_evaluator(scope);
        partial_evaluator.run();
    });

    // Eat all pe_known calls
    eval_intrinsics(world, [&] (auto callee, auto continuation) {
        if (callee->intrinsic() == Intrinsic::PeKnown) {
            eat_pe_known(continuation, false);
            return true;
        }
        return false;
    });

    world.cleanup();

    // Eat all other intrinsics
    eval_intrinsics(world, [&] (auto callee, auto continuation) {
        return eat_intrinsic(callee->intrinsic(), continuation, false);
    });
}

void partial_evaluation(World& world) {
    world.cleanup();
    VLOG_SCOPE(eval(world));

    world.mark_pe_done();
    world.cleanup();
}

//------------------------------------------------------------------------------

}
