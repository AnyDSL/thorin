#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

class Inliner : public Pass {
public:
    Inliner(PassMgr& mgr)
        : Pass(mgr)
    {
        states_.emplace_back();
    }

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

private:
    enum Lattice { Bottom, Inlined_Once, Dont_Inline };

    struct Info {
        Info() = default;
        Info(Lattice lattice, size_t undo)
            : lattice(lattice)
            , undo(undo)
        {}

        unsigned lattice :  4;
        unsigned undo    : 28;
    };

    using State = LamMap<Info>;

    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    Info& info(Lam* lam) { return cur_state().emplace(lam, Info(Lattice::Bottom, mgr().num_states())).first->second;  }
    void new_state() override { states_.emplace_back(cur_state()); }
    void undo(size_t u) override { states_.resize(u); }

    std::deque<LamMap<Info>> states_;
};

}

#endif
