#ifndef THORIN_PASS_BETA_ETA_CONV_H
#define THORIN_PASS_BETA_ETA_CONV_H

#include "thorin/pass/pass.h"

namespace thorin {

/**
 * Performs β- and η-conversion.
 * It uses the following strategy:
 * * β-reduction of <code>f e</code>happens if <code>f</code> only occurs exactly once in the program in callee position.
 * * η-reduction:
 *      * <code>λx.e x -> e</code>, whenever <code>x</code> does not appear free in <code>e</code> (always).
 *      * <code>f -> λx.f x</code>, if <code>f</code> is a @p Lam that does not appear in callee position and
 *          * occurs either more than once in non-callee position, or
 *          * appears at least once in calle position as well.
 *       This rule is a generalization of critical edge elimination.
 *       It gives other @p Pass%es such as @p SSAConstr the opportunity to change <code>f</code>'s signature (e.g. adding or removing params).
 *       The exact rules when which rule is triggered is best understood by the underlying lattice:
 * @code
 *          Eta_Wrap (η)                <-+
 *            /    \                      |- lam2eta_
 * Callee_N (-)     \                   <-+
 *           |      Non_Callee_1     (-) <-+
 * Callee_1 (β)     /                   <--- LamSet
 *            \    /
 *              Bot                     <--- not in any map
 * @endcode
 */
class BetaEtaConv : public Pass<BetaEtaConv> {
public:
    BetaEtaConv(PassMan& man, size_t index)
        : Pass(man, index, "reduction")
    {}

    enum class Lattice { Bot, Callee_1, Non_Callee_1, Callee_N, Eta_Wrap };
    enum class Loc { Bot, Callee_1, Non_Callee_1 };
    enum class Glob : uint16_t { Callee_N, Eta_Wrap };

    const Def* rewrite(Def*, const Def*) override;
    undo_t analyze(Def*, const Def*) override;
    Lattice lattice(Lam*);

    using Data = std::tuple<LamMap<Loc>>;

private:
    bool is_candidate(Lam* lam) { return !ignore(lam) && !man().is_tainted(lam); }

    LamSet keep_;
    LamMap<TaggedPtr<Lam, Glob>> lam2eta_;
};

}

#endif
