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

    enum Lattice { SSA, Keep };

    struct SlotInfo {
        SlotInfo() = default;
        SlotInfo(size_t undo)
            : lattice(Lattice::SSA)
            , undo(undo)
        {}

        unsigned lattice :  2;
        unsigned undo    : 30;
    };

    struct LamInfo {
        LamInfo() = default;
        LamInfo(size_t undo)
            : lattice(Lattice::SSA)
            , undo(undo)
        {}

        GIDMap<const Slot*, const Def*> slot2val;
        LamSet preds;
        std::vector<const Slot*> slots;
        Lam* new_lam = nullptr;
        unsigned lattice    :  2;
        unsigned undo       : 30;
    };

    using Slot2Info = GIDMap<const Slot*, SlotInfo>;
    using Lam2Info  = LamMap<LamInfo>;
    using Lam2Lam   = LamMap<Lam*>;
    using State     = std::tuple<Slot2Info, Lam2Info, Lam2Lam>;

private:
    const Def* get_val(Lam*, const Slot*);
    const Def* get_val(const Slot* slot) { return get_val(man().cur_lam(), slot); }
    void set_val(Lam*, const Slot*, const Def*);
    void set_val(const Slot* slot, const Def* def) { return set_val(man().cur_lam(), slot, def); }

    auto& slot2info(const Slot* slot) { return get<Slot2Info>(slot, SlotInfo(man().cur_state_id())); }
    auto& lam2info (Lam* lam)         { return get<Lam2Info> (lam,   LamInfo(man().cur_state_id())); }
    auto& lam2lam  (Lam* lam)         { return get<Lam2Lam>  (lam); }
};

}

#endif
