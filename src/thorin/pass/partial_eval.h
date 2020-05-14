#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public PassBase {
public:
    PartialEval(PassMan& man, size_t index)
        : PassBase(man, index, "partial_eval")
    {}

    const Def* rewrite(Def*, const Def*) override;
};

}

#endif
