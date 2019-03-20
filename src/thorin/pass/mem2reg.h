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

        SlotInfo()
            : lam(nullptr)
            , lattice(SSA)
            , undo(0x7fffffff)
        {}
        SlotInfo(Lam* lam, size_t undo)
            : lam(lam)
            , lattice(SSA)
            , undo(undo)
        {}

        Lam* lam;
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

        GIDMap<const Slot*, const Def*> slot2val;
        std::vector<const Slot*> slots;
        Lam* pred = nullptr;
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
    const Def* set_val(Lam*, const Slot*, const Def*);
    const Def* set_val(const Slot* slot, const Def* def) { return set_val(man().cur_lam(), slot, def); }
    const Def* virtual_phi(Lam*, const Slot*);

    auto& slot2info(const Slot* slot, SlotInfo init) { return get<Slot2Info>(slot, std::move(init)); }
    auto& slot2info(const Slot* slot) { return get<Slot2Info>(slot); }
    auto& lam2info (Lam* lam)         { return get<Lam2Info> (lam,   LamInfo(man().cur_state_id())); }
    auto& new2old  (Lam* lam)         { return get<Lam2Lam>  (lam); }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old(new_lam))
            return old_lam;
        return new_lam;
    }
};

}

#endif
