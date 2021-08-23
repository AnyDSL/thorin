#ifndef THORIN_PASS_FP_ETA_CONV_H
#define THORIN_PASS_FP_ETA_CONV_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Performs η-conversion.
 * It uses the following strategy:
 *  * η-reduction: <code>λx.e x -> e</code>, whenever <code>x</code> does (optimistically) not appear free in <code>e</code>.
 *  * η-expansion: <code>f -> λx.f x</code>, if <code>f</code> is a @p Lam with more than one user and does not appear in callee position.
 * This rule is a generalization of critical edge elimination.
 * It gives other @p Pass%es such as @p SSAConstr the opportunity to change <code>f</code>'s signature
 * (e.g. adding or removing @p Var%s).
 * @code
 *       expand_                <-- η-expand non-callee as occurs more than once; don't η-reduce the wrapper again.
 *        /   \
 *  Callee     Non_Callee_1     <-- Multiple callees XOR exactly one non-callee are okay.
 *        \   /
 *     irreducible_             <-- η-reduction not possible as we stumbled upon a Var.
 *          |
 *        Reduce                <-- η-reduction performed.
 *          |
 *          ⊥                   <-- Never seen.
 * @endcode
 */
class EtaConv : public FPPass<EtaConv, LamSet, LamSet> {
public:
    EtaConv(PassMan& man)
        : FPPass(man, "eta_conv")
    {}

    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

    enum : size_t { Reduce, Non_Callee_1 };

private:
    LamSet expand_;
    LamSet callee_;
    LamSet irreducible_;
    Def2Def def2exp_;
};

}

#endif
