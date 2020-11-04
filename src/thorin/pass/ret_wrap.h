#ifndef THORIN_PASS_RET_WRAP_H
#define THORIN_PASS_RET_WRAP_H

#include "thorin/pass/pass.h"

namespace thorin {

class RetWrap : public PassBase {
public:
    RetWrap(PassMan& man, size_t index)
        : PassBase(man, index, "ret_wrap")
    {}

    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*) override;

private:
    Def2Def old2new_;
    LamSet ret_conts_;
};

}

#endif

