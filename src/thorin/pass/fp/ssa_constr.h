#ifndef THORIN_PASS_FP_SSA_CONSTR_H
#define THORIN_PASS_FP_SSA_CONSTR_H

#include <map>
#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

/**
 * SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
 * This is loosely based upon:
 * "Simple and Efficient Construction of Static Single Assignment Form"
 * by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau. <br>
 * Depends on: @p EtaConv.
 */
class SSAConstr : public FPPass<SSAConstr> {
public:
    SSAConstr(PassMan& man, size_t index)
        : FPPass(man, "ssa_constr", index)
    {}

    enum : flags_t { Sloxy, Phixy, Traxy };

    struct V {
        GIDSet<const Proxy*> writable;
        Lam* pred = nullptr;
    };

    struct E {
        GIDMap<const Proxy*, const Def*> sloxy2val;
    };

    using Visit = std::map<Lam*, V, GIDLt<Lam*>>;
    using Enter = std::map<Lam*, E, GIDLt<Lam*>>;
    using Data = std::tuple<Visit, Enter>;

private:
    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;
    virtual undo_t analyze(Def* cur_nom) override;
    undo_t analyze(Def*, const Def*) override;

    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    undo_t join(Lam* cur_lam, Lam* lam, bool);
    const Def* mem2phi(Lam*, const App*, Lam*);

    size_t slot_id_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phixys_; ///< Contains the @p Phixy%s to add to @c mem_lam to build the @c phi_lam.
    GIDSet<const Proxy*> keep_;                                      ///< Contains @p Sloxy%s we want to keep.
    LamSet preds_n_;
    Lam2Lam mem2phi_;
    DefSet analyzed_;
};

}

#endif
