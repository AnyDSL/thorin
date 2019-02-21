#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include "thorin/pass/pass.h"

namespace thorin {

class Mem2Reg : public Pass {
public:
    Mem2Reg(PassMgr& mgr)
        : Pass(mgr)
    {
        states_.emplace_back();
    }

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;
    void new_state() override { states_.emplace_back(); }
    void undo(size_t u) override { states_.resize(u); }

private:
    enum Lattice { Bottom, SSA, Keep };

    struct Info {
        Info() = default;
        Info(Lattice lattice, size_t undo)
            : lattice(lattice)
            , undo(undo)
        {}

        unsigned lattice :  4;
        unsigned undo    : 28;
    };

    struct State {
        GIDMap<const Slot*, Info> slot2info;
        LamMap<Array<Lam*>> lam2preds;
    };

    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    Info& info(const Slot*);
    Array<Lam*>& lam2preds(Lam*);

    std::deque<State> states_;
};

}

#endif
