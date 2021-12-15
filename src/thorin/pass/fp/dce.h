#ifndef THORIN_PASS_FP_DCE_H
#define THORIN_PASS_FP_DCE_H

#include "thorin/pass/pass.h"
#include "thorin/util/bitset.h"

namespace thorin {

class BetaRed;
class EtaExp;

/// Dead Code Elimination.
class DCE : public FPPass<DCE, Lam> {
public:
    DCE(PassMan& man, BetaRed* beta_red, EtaExp* eta_exp)
        : FPPass(man, "dce")
        , beta_red_(beta_red)
        , eta_exp_(eta_exp)
    {}

    using Data = LamSet;

private:
    /// @name PassMan hooks
    //@{
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Proxy*) override;
    //@}

    const Def* var2dead(const App*, Lam*);

    BetaRed* beta_red_;
    EtaExp* eta_exp_;
    LamMap<std::pair<Lam*, BitSet>> var2dead_;
    DefSet keep_;
};

}

#endif
