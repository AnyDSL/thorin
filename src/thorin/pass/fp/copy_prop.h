#ifndef THORIN_PASS_FP_COPY_PROP_H
#define THORIN_PASS_FP_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

class BetaRed;
class EtaExp;

/// This @p FPPass is similar to sparse conditional constant propagation (SCCP).
/// However, this optmization also works on all @p Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
/// What is more, this optimization will also propagate arbitrary @p Def%s and not only constants.
class CopyProp : public FPPass<CopyProp, Lam> {
public:
    CopyProp(PassMan& man, BetaRed* beta_red, EtaExp* eta_exp)
        : FPPass(man, "copy_prop")
        , beta_red_(beta_red)
        , eta_exp_(eta_exp)
    {}

    using Data = LamMap<DefVec>;

private:
    /// @name PassMan hooks
    //@{
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Proxy*) override;
    //@}

    const Def* var2prop(const App*, Lam*);

    BetaRed* beta_red_;
    EtaExp* eta_exp_;
    LamMap<std::pair<Lam*, DefVec>> var2prop_;
    DefSet keep_;
};

}

#endif
