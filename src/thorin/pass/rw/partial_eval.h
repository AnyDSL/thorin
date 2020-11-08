#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public RWPass {
public:
    PartialEval(PassMan& man, size_t index)
        : RWPass(man, index, "partial_eval")
    {}

    const Def* rewrite(Def*, const Def*) override;
};

}

#endif
