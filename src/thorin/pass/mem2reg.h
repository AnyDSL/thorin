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

    using Slot2Info    = GIDMap<const Slot*, Info> ;
    using Lam2Preds    = LamMap<Array<Lam*>>;
    using Lam2Slot2Val = LamMap<std::unique_ptr<GIDMap<const Slot*, const Def*>>>;
    using State        = std::tuple<Slot2Info, Lam2Preds, Lam2Slot2Val>;

    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    auto& slot2info(const Slot* slot) { return get<Mem2Reg, Slot2Info>   (slot, std::move(Info(Lattice::SSA, mgr().num_states()))); }
    auto& lam2preds   (Lam* lam)      { return get<Mem2Reg, Lam2Preds>   (lam,  std::move(Array<Lam*>())); }
    auto& lam2slot2val(Lam* lam)      { return get<Mem2Reg, Lam2Slot2Val>(lam,  std::move(std::make_unique<GIDMap<const Slot*, const Def*>>())); }

    std::deque<State> states_;

    friend class Pass;
};

}

#endif
