#ifndef THORIN_PASS_FP_COPY_PROP_H
#define THORIN_PASS_FP_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

class EtaExp;

/// This @p FPPass is similar to sparse conditional constant propagation (SCCP).
/// However, this optmization also works on all @p Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
/// What is more, this optimization will also propagate arbitrary @p Def%s and not only constants.
class CopyProp : public FPPass<CopyProp, Lam> {
public:
    CopyProp(PassMan& man, EtaExp* eta_exp)
        : FPPass(man, "copy_prop")
        , eta_exp_(eta_exp)
    {}

    using Data = LamMap<DefVec>;

private:
    /// @name PassMan hooks
    //@{
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Proxy*) override;
    undo_t analyze(const Def*) override;
    //@}

    const Def* var2prop(const App*, Lam*);

    LamMap<std::pair<Lam*, DefVec>> var2prop_;
    DefSet keep_;
    EtaExp* eta_exp_;
};

}

#endif
