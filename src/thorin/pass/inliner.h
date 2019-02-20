#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass {
public:
    Inliner(PassMgr& mgr)
        : Pass(mgr)
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
