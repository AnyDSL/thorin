#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

static void verify_top_level(World& world) {
    Scope::for_each(world, [&] (const Scope& scope) {
        if (scope.has_free_params()) {
            for (auto param : scope.free_params())
                ELOG("top-level continuation '{}' got free param '{}' belonging to continuation {}", scope.entry(), param, param->continuation());
            ELOG("here: {}", scope.entry());
        }
    });
}

#if 0

class Cycles {
public:
    enum Color {
        Gray, Black
    };

    Cycles(World& world)
        : world_(world)
    {
        size_t num = world.primops().size();
        for (auto continuation : world.continuations())
            num += 1 + continuation->num_params();
        def2color_.rehash(round_to_power_of_2(num));
    }

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
    if (def2color_.emplace(continuation, Gray).second) {
        ParamSet params;
        for (auto op : continuation->ops())
            analyze(params, continuation, op);

        for (auto param : params) {
            if (def2color_.emplace(param, Gray).second) {
                analyze_call(param->continuation());
                def2color_[param] = Black;
            }
        }

        def2color_[continuation] = Black;
    } else
        assertf(def2color_[continuation] != Gray, "detected cycle: '{}'", continuation);
}

void Cycles::analyze(ParamSet& params, const Continuation* continuation, const Def* def) {
    if (auto primop = def->isa<PrimOp>()) {
        if (def2color_.emplace(def, Black).second) {
            for (auto op : primop->ops())
                analyze(params, continuation, op);
        }
    } else if (auto param = def->isa<Param>()) {
        if (param->continuation() != continuation) {
            auto i = def2color_.find(param);
            if (i != def2color_.end())
                assertf(i->second != Gray, "detected cycle induced by parameter: '{}'", param);
            else
                params.emplace(param);
        }
    }
}
#endif

void verify(World& world) {
    verify_top_level(world);
    //Cycles cycles(world);
    //cycles.run();
}

}
