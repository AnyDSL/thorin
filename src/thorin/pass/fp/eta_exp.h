#ifndef THORIN_PASS_FP_ETA_EXP_H
#define THORIN_PASS_FP_ETA_EXP_H

#include "thorin/pass/pass.h"

namespace thorin {

class EtaRed;

/// Performs η-expansion:
/// <code>f -> λx.f x</code>, if <code>f</code> is a @p Lam with more than one user and does not appear in callee position.
/// This rule is a generalization of critical edge elimination.
/// It gives other @p Pass%es such as @p SSAConstr the opportunity to change <code>f</code>'s signature
/// (e.g. adding or removing @p Var%s).
class EtaExp : public FPPass<EtaExp, Lam> {
public:
    EtaExp(PassMan& man, EtaRed* eta_red)
        : FPPass(man, "eta_exp")
        , eta_red_(eta_red)
    {}

    /*
    @code
         expand_                <-- η-expand non-callee as it occurs more than once; don't η-reduce the wrapper again.
          /   \
    Callee     Non_Callee_1     <-- Multiple callees XOR exactly one non-callee are okay.
          \   /
           Bot                  <-- Never seen.
    @endcode
    */
    enum Lattice : bool { Callee, Non_Callee_1 };
    static const char* lattice2str(Lattice l) { return l == Callee ? "Callee" : "Non_Callee_1"; }

    using Data = LamMap<std::pair<Lattice, undo_t>>;
    undo_t join(Lam*, Lattice);

private:
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

    EtaRed* eta_red_;
    LamSet expand_;
    Def2Def def2exp_;
    LamMap<std::pair<Lam*, const Def*>> wrap2subst_;
};

}

#endif
