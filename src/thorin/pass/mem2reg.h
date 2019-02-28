#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include "thorin/pass/pass.h"

namespace thorin {

class Mem2Reg : public Pass<Mem2Reg> {
public:
    Mem2Reg(PassMgr& mgr, size_t pass_index)
        : Pass(mgr, pass_index)
    {}

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

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

    using Slot2Info    = GIDMap<const Slot*, Info>;
    using Lam2Preds    = LamMap<LamSet>;
    using Lam2Slot2Val = LamMap<std::unique_ptr<GIDMap<const Slot*, const Def*>>>;
    using State        = std::tuple<Slot2Info, Lam2Preds, Lam2Slot2Val>;

private:
    const Def* get_val(Lam*, const Slot*);
    void set_val(Lam*, const Slot*, const Def*);

    auto& slot2info(const Slot* slot) { return get<Slot2Info>   (slot, Info(Lattice::Bottom, mgr().state_id())); }
    auto& lam2slot2val(Lam* lam)      { return get<Lam2Slot2Val>(lam,  std::make_unique<GIDMap<const Slot*, const Def*>>()); }
    auto& lam2preds   (Lam* lam)      { return get<Lam2Preds>   (lam); }
};

}

#endif
