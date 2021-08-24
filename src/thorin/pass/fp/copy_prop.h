#if 0
#ifndef THORIN_PASS_FP_COPY_PROP_H
#define THORIN_PASS_FP_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

/// This @p FPPass is similar to sparse conditional constant propagation (SCCP) but also propagates arbitrary values through @p Var%s.
/// However, this optmization also works on all @p Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
/// What is more, this optimization will also propagate arbitrary @p Def%s and not only constants. <br>
/// Depends on: @p EtaConv.
class CopyProp : public FPPass<CopyProp, LamMap<std::vector<const Def*>>> {
public:
    CopyProp(PassMan& man)
        : FPPass(man, "copy_prop")
    {}

private:
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

    Lam2Lam var2prop_;
    DefSet keep_;
};

}

#endif
#endif
