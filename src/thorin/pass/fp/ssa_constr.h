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
    SSAConstr(PassMan& man)
        : FPPass(man, "ssa_constr")
    {}

    enum : flags_t { Sloxy, Phixy, Traxy };

    struct Info {
        Lam* pred = nullptr;
        GIDSet<const Proxy*> writable;
    };

    using Lam2Info = std::map<Lam*, Info, GIDLt<Lam*>>;
    using Data = std::tuple<Lam2Info>;

private:
    void enter() override;
    const Def* rewrite(const Def*) override;
    undo_t analyze() override;
    undo_t analyze(const Def*) override;

    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    undo_t join(Lam* cur_lam, Lam* lam, bool);
    const Def* mem2phi(Lam*, const App*, Lam*);

    std::map<Lam*, GIDMap<const Proxy*, const Def*>, GIDLt<Lam*>> lam2sloxy2val_;
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2phixys_; ///< Contains the @p Phixy%s to add to @c mem_lam to build the @c phi_lam.
    GIDSet<const Proxy*> keep_;                                      ///< Contains @p Sloxy%s we want to keep.
    LamSet preds_n_;
    Lam2Lam mem2phi_;
    DefSet analyzed_;
};

}

#endif
