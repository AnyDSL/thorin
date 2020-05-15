#ifndef THORIN_PASS_SSA_CONSTR_H
#define THORIN_PASS_SSA_CONST_RH

#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau.
 */
class SSAConstr : public Pass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : Pass(man, index, "ssa_constr")
    {}

    void inspect(Def*, Def*) override;
    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    struct Info {
        enum Lattice { Preds0, Preds1, PredsN, Keep };

        Info() = default;
        Info(size_t undo)
            : lattice(Preds0)
            , undo(undo)
        {}

        GIDMap<const Analyze*, const Def*> proxy2val;
        GIDSet<const Analyze*> writable;
        Lam* pred = nullptr;
        Lam* new_lam = nullptr;
        unsigned num_slots = 0;
        unsigned lattice :  2;
        unsigned undo    : 30;
    };

    using Lam2Info = LamMap<Info>;
    using State    = std::tuple<Lam2Info>;

private:
    const Analyze* isa_proxy(const Def*);
    const Analyze* isa_virtual_phi(const Def*);
    const Def* get_val(Lam*, const Analyze*);
    const Def* set_val(Lam*, const Analyze*, const Def*);

    auto& lam2info(Lam* lam) { return get<Lam2Info>(lam, Info(man().cur_state_id())).first->second; }
    Lam* original(Lam* new_lam) {
        if (auto old_lam = new2old_.lookup(new_lam)) return *old_lam;
        return new_lam;
    }

    LamMap<Lam*> new2old_;
    LamMap<std::set<const Analyze*, GIDLt<const Analyze*>>> lam2phis_;
    DefSet keep_;
    LamSet preds_n_;
};

}

#endif
