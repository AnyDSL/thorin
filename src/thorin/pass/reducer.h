#ifndef THORIN_PASS_INLINER_H
#define THORIN_PASS_INLINER_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Performs β- and η-reduction.
 * * β-reduction happens optimistically if a @p Lam only occurs exactly once in callee position.
 * * η-reduction: (TODO)
 *     * <code>λx.e x -> f</code>, whenever <code>x</code> does not appear free in <code>e</code>.
 *     * <code>f -> λx.f x</code>, if <code>f</code> is a @p Lam that appears in callee and some other position.
 *       This rule has the effect of critical edge elimination and gives other @p Pass%es such as @p SSAConstr the opportunity
 *       to change <code>f</code>'s signature (e.g. adding or removing parameters).
 */
class Reducer : public Pass<Reducer> {
public:
    Reducer(PassMan& man, size_t index)
        : Pass(man, index, "reducer")
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
