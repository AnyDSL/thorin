#ifndef THORIN_PASS_RET_WRAP_H
#define THORIN_PASS_RET_WRAP_H

#include "thorin/pass/pass.h"

namespace thorin {

class RetWrap : public RWPass {
public:
    RetWrap(PassMan& man)
        : RWPass(man, "ret_wrap")
    {}

    void enter(Def*) override;
    const Def* rewrite(Def*, const Def*, const Def*, Defs, const Def*) override;

private:
    Def2Def old2new_;
    LamSet ret_conts_;
};

}

#endif

