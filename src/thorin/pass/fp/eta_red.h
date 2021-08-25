#ifndef THORIN_PASS_FP_ETA_RED_H
#define THORIN_PASS_FP_ETA_RED_H

#include "thorin/pass/pass.h"

namespace thorin {

/// Performs η-reduction:
/// <code>λx.e x -> e</code>, whenever <code>x</code> does (optimistically) not appear free in <code>e</code>.
class EtaRed : public FPPass<EtaRed> {
public:
    EtaRed(PassMan& man)
        : FPPass(man, "eta_red")
    {}

    enum Lattice {
        Bot,         ///< Never seen.
        Reduce,      ///< η-reduction performed.
        Irreducible, ///< η-reduction not possible as we stumbled upon a Var.
    };

    using Data = LamMap<std::pair<Lattice, undo_t>>;

private:
    const Def* rewrite(const Def*) override;
    undo_t analyze(const Var*) override;

    LamSet irreducible_;

    friend class EtaExp;
};

}

#endif
