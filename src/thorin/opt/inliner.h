#ifndef THORIN_OPT_INLINER_H
#define THORIN_OPT_INLINER_H

#include "thorin/opt/optimizer.h"

namespace thorin {

class Inliner : public Optimization {
public:
    Inliner(Optimizer& optimizer)
        : Optimization(optimizer)
    {}

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

private:
    enum class State : uint8_t { Bottom, Inlined_Once, Dont_Inline };
    State& state(Lam* lam) { return state_.emplace(lam, State::Bottom).first->second;  }

    LamMap<State> state_;
};

}

#endif
