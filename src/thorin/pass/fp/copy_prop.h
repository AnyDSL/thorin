#ifndef THORIN_PASS_FP_COPY_PROP_H
#define THORIN_PASS_FP_COPY_PROP_H

#include "thorin/pass/pass.h"

namespace thorin {

/// This @p FPPass is similar to sparse conditional constant propagation (SCCP) but also propagates arbitrary values through @p Param%s.
/// However, this optmization also works on all @p Lam%s alike and does not only consider basic blocks as opposed to traditional SCCP.
/// What is more, this optimization will also propagate arbitrary @p Def%s and not only constants.
class CopyProp : public FPPass<CopyProp> {
public:
    CopyProp(PassMan& man, size_t index)
        : FPPass(man, index, "copy_prop")
    {}

    using Args = std::vector<const Def*>;
    using Data = std::tuple<LamMap<Args>>;

private:
    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    Lam2Lam param2prop_;
    DefSet keep_;
};

}

#endif
