#ifndef THORIN_PASS_REDUCTION_H
#define THORIN_PASS_REDUCTION_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Optimistically performs β- and η-reduction.
 * It uses the following strategy:
 * * β-reduction of <code>f e</code>happens if <code>f</code> only occurs exactly once in the program in callee position.
 * * η-reduction: (TODO)
 *     * <code>λx.e x -> e</code>, whenever <code>x</code> does not appear free in <code>e</code>.
 *     * <code>f -> λx.f x</code>, if <code>f</code> is a @p Lam that appears in callee and some other position.
 *       This rule is a generalization of critical edge elimination.
 *       It gives other @p Pass%es such as @p SSAConstr the opportunity to change <code>f</code>'s signature (e.g. adding or removing parameters).
 */
class Reduction : public Pass<Reduction> {
public:
    Reduction(PassMan& man, size_t index)
        : Pass(man, index, "reduction")
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
