#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public PassBase {
public:
    PartialEval(PassMgr& mgr, size_t id)
        : PassBase(mgr, id)
    {}

    const Def* rewrite(const Def*) override;
};

}

#endif
