#ifndef THORIN_PASS_RET_WRAP_H
#define THORIN_PASS_RET_WRAP_H

#include "thorin/pass/pass.h"

namespace thorin {

class RetWrap : public RWPass<Lam> {
public:
    RetWrap(PassMan& man)
        : RWPass(man, "ret_wrap")
    {}

    void enter() override;
};

}

#endif

