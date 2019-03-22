#ifndef THORIN_PASS_MEM2REG_H
#define THORIN_PASS_MEM2REG_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau
 */
class Mem2Reg : public Pass<Mem2Reg> {
public:
    Mem2Reg(PassMan& man, size_t id)
        : Pass(man, id)
    {}

    const Def* rewrite(const Def*) override;
    void inspect(Def*) override;
    void enter(Def*) override;
    void analyze(const Def*) override;

    struct SlotInfo {
        enum Lattice { SSA, Keep_Slot };

        SlotInfo() = default;
        SlotInfo(size_t undo)
            : lattice(SSA)
            , undo(undo)
        {}

        unsigned lattice :  1;
        unsigned undo    : 31;
    };

    struct LamInfo {
        enum Lattice { Preds0, Preds1, PredsN, Keep_Lam };

        LamInfo() = default;
        LamInfo(size_t undo)
            : lattice(Preds0)
            , undo(undo)
        {}

        std::vector<const Slot*> slots;
        Lam* pred = nullptr;
        Lam* new_lam = nullptr;
        unsigned lattice    :  2;
        unsigned undo       : 30;
    };

    using Slot2Info    = GIDMap<const Slot*, SlotInfo>;
    using Lam2Info     = LamMap<LamInfo>;
    using Lam2Lam      = LamMap<Lam*>;
    using Mem2Slot2Val = DefMap<GIDMap<const Slot*, const Def*>>;
    using State        = std::tuple<Slot2Info, Lam2Info, Mem2Slot2Val, Lam2Lam>;

private:
    const Slot* is_ssa_slot(const Def*);
    const Def* get_val(const Def*, const Slot*);
    const Def* set_val(const Def*, const Slot*, const Def*);

    auto& slot2info   (const Slot* slot) { return get<Slot2Info>(slot, SlotInfo(man().cur_state_id())); }
    auto& lam2info    (Lam* lam)         { return get<Lam2Info> (lam,   LamInfo(man().cur_state_id())); }
    auto& mem2slot2val(const Def* mem)   { return get<Mem2Slot2Val>(mem); }
    auto& new2old     (Lam* lam)         { return get<Lam2Lam>  (lam); }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old(new_lam)) return old_lam;
        return new_lam;
    }
};

}

#endif
