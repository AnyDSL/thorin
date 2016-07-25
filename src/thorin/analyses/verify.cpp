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
    void analyze(ParamSet& params, const Continuation*, const Def*);

private:
    World& world_;
    DefMap<Color> def2color_;
};

void Cycles::run() {
    for (auto continuation : world().continuations())
        analyze_call(continuation);
}

void Cycles::analyze_call(const Continuation* continuation) {
    auto p = def2color_.emplace(continuation, Black);
    if (p.second) {
        ParamSet params;
        for (auto op : continuation->ops())
            analyze(params, continuation, op);

        for (auto param : params) {
            auto p = def2color_.emplace(param, Gray);
            assert(p.second);
            analyze_call(param->continuation());
            def2color_[param] = Black;
        }
    }
}

void Cycles::analyze(ParamSet& params, const Continuation* cont, const Def* def) {
    if (def->isa<Continuation>())
        return;

    if (auto param = def->isa<Param>()) {
        if (param->continuation() != cont) {
            auto i = def2color_.find(param);
            if (i != def2color_.end()) {
                auto color = i->second;
                switch (color) {
                    case Gray:
                        ELOG("alles karpott: %", param);
                    case Black:
                        return;
                }
                params.emplace(param);
            }
        }
        return;
    }

    if (auto primop = def->isa<PrimOp>()) {
        auto p = def2color_.emplace(def, Black);
        if (p.second) {
            for (auto op : primop->ops())
                analyze(params, cont, op);
        } 
    }
}

void verify(World& world) {
    verify_calls(world);
    Cycles cycles(world);
    cycles.run();
}

}
