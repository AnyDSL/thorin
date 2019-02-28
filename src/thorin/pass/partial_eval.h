#ifndef THORIN_PASS_PARTIAL_EVAL_H
#define THORIN_PASS_PARTIAL_EVAL_H

#include "thorin/pass/pass.h"

namespace thorin {

class PartialEval : public Pass<PartialEval> {
public:
    PartialEval(PassMgr& mgr, size_t pass_index)
        : Pass(mgr, pass_index)
    {}

    const Def* rewrite(const Def*) override;

    struct State {};
};

}

#endif
