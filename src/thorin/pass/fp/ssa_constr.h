#ifndef THORIN_PASS_FP_SSA_CONSTR_H
#define THORIN_PASS_FP_SSA_CONSTR_H

#include <map>
#include <set>

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

class EtaExp;

/// SSA construction algorithm that promotes @p Slot%s, @p Load%s, and @p Store%s to SSA values.
/// This is loosely based upon:
/// "Simple and Efficient Construction of Static Single Assignment Form"
/// by Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau. <br>
class SSAConstr : public FPPass<SSAConstr, Lam> {
public:
    SSAConstr(PassMan& man, EtaExp* eta_exp)
        : FPPass(man, "ssa_constr")
        , eta_exp_(eta_exp)
    {}

    enum : flags_t { Phixy, Sloxy, Traxy };

    struct Info {
        Lam* pred = nullptr;
        GIDSet<const Proxy*> writable;
    };

    using Data = std::map<Lam*, Info, GIDLt<Lam*>>;

private:
    /// @name PassMan hooks
    //@{
    void enter() override;
    const Def* rewrite(const Proxy*) override;
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Proxy*) override;
    undo_t analyze(const Def*) override;
    //@}

    /// @name SSA construction helpers - see paper
    //@{
    const Def* get_val(Lam*, const Proxy*);
    const Def* set_val(Lam*, const Proxy*, const Def*);
    const Def* mem2phi(const App*, Lam*);
    //@}

    EtaExp* eta_exp_;
    Lam2Lam mem2phi_;

    /// Value numbering table.
    std::map<Lam*, GIDMap<const Proxy*, const Def*>, GIDLt<Lam*>> lam2sloxy2val_;

    /// Contains the @p Sloxy%s that we need to install as phi in a @c mem_lam to build the @c phi_lam.
    LamMap<std::set<const Proxy*, GIDLt<const Proxy*>>> lam2sloxys_;

    /// Contains @p Sloxy%s we have to keep.
    GIDSet<const Proxy*> keep_;
};

}

#endif
