#ifndef THORIN_PASS_SSA_CONSTR_H
#define THORIN_PASS_SSA_CONST_RH

#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/// SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
/// This is loosely based upon:
/// "Simple and Efficient Construction of Static Single Assignment Form"
/// by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau.
class SSAConstr : public Pass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : Pass(man, index, "ssa_constr")
    {}

    struct Visit {
        Lam* pred = nullptr;
        enum { Preds0, Preds1 } preds;
        Lam* phi_lam = nullptr;
    };

    struct Enter {
        GIDMap<const Proxy*, const Def*> sloxy2val;
        GIDSet<const Proxy*> writable;
        uint32_t num_slots;
    };

    using State = std::tuple<LamMap<Visit>, LamMap<Enter>>;

private:
    Def* mem2phi(Def*);
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    const Proxy* isa_sloxy(const Def*);
    const Proxy* isa_phixy(const Def*);
    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);

    template<class T> // T = Visit or Enter
    std::pair<T&, undo_t> get(Lam* lam) { auto [i, undo, ins] = insert<LamMap<T>>(lam); return {i->second, undo}; }
    Lam* phi2mem(Lam* phi_lam) { auto mem_lam = phi2mem_.lookup(phi_lam); return mem_lam ? *mem_lam : nullptr; }
    Lam* lam2mem(Lam* lam) { auto mem_lam = phi2mem(lam); return mem_lam ? mem_lam : lam; }

    LamMap<Lam*> phi2mem_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phis_; ///< Contains the phis we have to add to the mem_lam to build the phi_lam.
    DefSet keep_;                                                  ///< Contains Lams as well as sloxys we want to keep.
    LamSet preds_n_;                                               ///< Contains Lams with more than one preds.
};

}

#endif
