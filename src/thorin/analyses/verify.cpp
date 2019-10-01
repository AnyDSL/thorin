#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

static void verify_top_level(World& world) {
    world.visit([&](const Scope& scope) {
        if (scope.has_free_params()) {
            for (auto param : scope.free_params())
                world.ELOG("top-level lam '{}' got free param '{}' belonging to lam {}", scope.entry(), param, param->lam());
            world.ELOG("here: {}", scope.entry());
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
        for (auto lam : world.lams())
            num += 1 + lam->num_params();
        def2color_.rehash(round_to_power_of_2(num));
    }

    World& world() { return world_; }
    void run();
    void analyze_call(const Lam*);
    void analyze(ParamSet& params, const Lam*, const Def*);

private:
    World& world_;
    DefMap<Color> def2color_;
};

void Cycles::run() {
    for (auto lam : world().lams())
        analyze_call(lam);
}

void Cycles::analyze_call(const Lam* lam) {
    if (def2color_.emplace(lam, Gray).second) {
        ParamSet params;
        for (auto op : lam->ops())
            analyze(params, lam, op);

        for (auto param : params) {
            if (def2color_.emplace(param, Gray).second) {
                analyze_call(param->lam());
                def2color_[param] = Black;
            }
        }

        def2color_[lam] = Black;
    } else
        assertf(def2color_[lam] != Gray, "detected cycle: '{}'", lam);
}

void Cycles::analyze(ParamSet& params, const Lam* lam, const Def* def) {
    if (auto primop = def->isa<PrimOp>()) {
        if (def2color_.emplace(def, Black).second) {
            for (auto op : primop->ops())
                analyze(params, lam, op);
        }
    } else if (auto param = def->isa<Param>()) {
        if (param->lam() != lam) {
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
