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
    const Def* get_val(Lam*, const Slot*);
    void set_val(Lam*, const Slot*, const Def*);

    enum Lattice { SSA, Keep };

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
        LamMap<std::unique_ptr<GIDMap<const Slot*, const Def*>>> lam2slot2val;
    };

    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    Info& info(const Slot*);
    ArrayRef<Lam*> lam2preds(Lam*);
    GIDMap<const Slot*, const Def*>& lam2slot2val(Lam*);

    std::deque<State> states_;
};

}

#endif
