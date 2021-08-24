#if 0
#ifndef THORIN_PASS_FP_SCALARIZE_H
#define THORIN_PASS_FP_SCALARIZE_H

#include "thorin/pass/pass.h"

namespace thorin {

class Scalerize : public FPPass<Scalerize> {
public:
    Scalerize(PassMan& man)
        : FPPass(man, "scalerize")
    {}

    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

private:
    DefSet keep_;
    Lam2Lam tup2sca_;
};

}

#endif
#endif
