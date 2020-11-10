#ifndef THORIN_PASS_FP_ETA_CONV_H
#define THORIN_PASS_FP_ETA_CONV_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Performs η-conversion.
 * It uses the following strategy:
 *  1. η-reduction: <code>λx.e x -> e</code>, whenever <code>x</code> does not appear free in <code>e</code> and does not contradict rule 2).
 *  2. η-expansion: <code>f -> λx.f x</code>, if
 *      1. <code>f</code> is a @p Lam that does not appear in callee position.
 *      2. <code>f</code>
 *      This rule is a generalization of critical edge elimination.
 *      It gives other @p Pass%es such as @p SSAConstr the opportunity to change <code>f</code>'s signature (e.g. adding or removing @p Param%s).
 */
class EtaConv : public FPPass<EtaConv> {
public:
    EtaConv(PassMan& man, size_t index)
        : FPPass(man, "eta_conv", index)
    {}

    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;

    using Data = std::tuple<LamSet>;

private:
    LamSet keep_;
    Def2Def def2eta_;
    DefSet wrappers_;
};

}

#endif
