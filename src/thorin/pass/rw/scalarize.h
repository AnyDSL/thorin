#ifndef THORIN_PASS_RW_SCALARIZE_H
#define THORIN_PASS_RW_SCALARIZE_H

#include "thorin/world.h"
#include "thorin/pass/pass.h"
#include "thorin/pass/fp/eta_exp.h"

namespace thorin {

/// Perform Scalarization (= Argument simplification), i.e.:
/// <code> f := λ (x_1:[T_1, T_2], .., x_n:T_n).E </code> will be transformed to
/// <code> f' := λ (y_1:T_1, y_2:T2, .. y_n:T_n).E[x_1\(y_1, y2); ..; x_n\y_n]</code> if
/// <code>f</code> appears in callee position only, see @p EtaExp.
/// It will not flatten nominal @p Sigma#s or @p Arr#s.
class Scalerize : public RWPass<Lam> {
public:
    Scalerize(PassMan& man, EtaExp* eta_exp)
        : RWPass(man, "scalerize")
        , eta_exp_(eta_exp)
    {}

    const Def* rewrite(const Def*) override;

private:
    bool should_expand(Lam *lam);
    Lam* make_scalar(Lam *lam);

    EtaExp* eta_exp_;
    Lam2Lam tup2sca_;
};

}

#endif
