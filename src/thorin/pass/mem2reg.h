#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include "thorin/pass/pass.h"

namespace thorin {

class Mem2Reg : public Pass<Mem2Reg> {
public:
    Mem2Reg(PassMan& man, size_t id)
        : Pass(man, id)
    {}

    Def* rewrite(Def*) override;
    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

    enum Lattice { SSA, Keep };

    struct SlotInfo {
        SlotInfo() = default;
        SlotInfo(Lattice lattice, size_t undo)
            : lattice(lattice)
            , undo(undo)
        {}

        unsigned lattice :  4;
        unsigned undo    : 28;
    };

    struct LamInfo {
        GIDMap<const Slot*, const Def*> slot2val;
        LamSet preds;
        std::vector<const Slot*> slots;
    };

    using Slot2Info = GIDMap<const Slot*, SlotInfo>;
    using Lam2Info  = LamMap<LamInfo>;
    using State     = std::tuple<Slot2Info, Lam2Info>;

private:
    const Def* get_val(Lam*, const Slot*);
    void set_val(Lam*, const Slot*, const Def*);

    auto& slot2info(const Slot* slot) { return get<Slot2Info>(slot, SlotInfo(Lattice::SSA, man().state_id())); }
    auto& lam2info (Lam* lam)         { return get<Lam2Info> (lam); }
};

}

#endif
