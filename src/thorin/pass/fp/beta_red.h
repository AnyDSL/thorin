#ifndef THORIN_PASS_FP_BETA_RED_H
#define THORIN_PASS_FP_BETA_RED_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Optimistically performs β-reduction (aka inlining).
 * β-reduction of <code>f e</code>happens if <code>f</code> only occurs exactly once in the program in callee position.
 */
class BetaRed : public FPPass<BetaRed> {
public:
    BetaRed(PassMan& man, size_t index)
        : FPPass(man, "beta_red", index)
    {}

    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    using Data = std::tuple<LamSet>;

private:
    bool is_candidate(Lam* lam) { return !ignore(lam) && !man().is_tainted(lam); }

    LamSet keep_;
};

}

#endif
