#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#if 0
#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public PassBase {
public:
    PartialEval(PassMan& man, size_t index)
        : PassBase(man, index)
    {}

    const Def* rewrite(const Def*) override;
};

}

#endif
#endif
