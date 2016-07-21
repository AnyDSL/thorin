#include <stack>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

static void verify_calls(World& world) {
    for (auto continuation : world.continuations()) {
        if (!continuation->empty())
            assert(continuation->callee_fn_type()->num_args() == continuation->arg_fn_type()->num_args() && "argument/parameter mismatch");
    }
}

#if 0
static void detect_cycles(World& world) {
    DefMap<Color> def2color;
    std::stack<const Def*> stack;

    for (auto continuation : world.continuations()) {
        for (auto param : continuation->params())
            def2color[param] = Black;
        for (auto op : continuation->ops()) {
            if (op->isa_continuation())
                continue;

            if (def2color.emplace(op, White).second) {
                stack.push(op);

                while (!stack.empty()) {
                    bool todo = false;
                    auto def = stack.top();
                    assert(def->isa<PrimOp>() || def->isa<Param>());

                    assert(def2color.contains(def));
                    auto& color = def2color.find(def)->second;

                    if (color == White) {
                        if (auto param = def->isa<Param>()) {
                            def = param->continuation();
                            for (auto p : param->continuation()->params())
                                def2color[p] = Black;
                        }

                        for (auto op : def->ops()) {
                            if (op->isa_continuation())
                                continue;

                            if (def2color.emplace(op, White).second) {
                                stack.push(op);
                                todo = true;
                                continue;
                            }

                            auto& color = def2color.find(op)->second;
                            if (color == White) {
                            } else if (color == Gray) {
                                ELOG("detected cycle at %", op);
                            } else { // color == Black
                            }
                        }
                    }

                    if (todo)
                        color = Gray;
                    else {
                        color = Black;
                        stack.pop();
                    }
                }
            }
        }
    }
}
#endif

class Cycles {
public:
    enum Color {
        Gray, Black
    };

    Cycles(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void run();
    void analyze_call(const Continuation*);
    void analyze(const Def*);

private:
    World& world_;
    DefMap<Color> def2color_;
};

void Cycles::run() {
    for (auto continuation : world().continuations())
        analyze_call(continuation);
}

void Cycles::analyze_call(const Continuation* continuation) {
    for (auto param : continuation->params()) {
        def2color_[param] = Black; // TODO too relaxed?
    }

    for (auto op : continuation->ops())
        analyze(op);
}

void Cycles::analyze(const Def* def) {
    if (def->isa<Continuation>())
        return;

    auto p = def2color_.emplace(def, Gray);
    auto& color = p.first->second;

    if (p.second) {
        if (auto primop = def->isa<PrimOp>()) {
            for (auto op : primop->ops())
                analyze(op);
        } else if (auto param = def->isa<Param>()) {
            analyze_call(param->continuation());
        } else if (def->isa<Continuation>())
            return;
    } else {
        if (def->isa<Param>() && color == Gray)
            WLOG("detected cycle at %", def);
        // Black: do nothing
    }

    color = Black;
}

void verify(World& world) {
    verify_calls(world);
    Cycles cycles(world);
    cycles.run();
}

}
