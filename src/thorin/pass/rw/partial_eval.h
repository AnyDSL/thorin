#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public RWPass {
public:
    PartialEval(PassMan& man)
        : RWPass(man, "partial_eval")
    {}

    const Def* rewrite(const Def*) override;
};

}

#endif
