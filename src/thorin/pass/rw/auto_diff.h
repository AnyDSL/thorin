#ifndef THORIN_PASS_RW_AUTO_DIFF_H
#define THORIN_PASS_RW_AUTO_DIFF_H

#include "thorin/pass/pass.h"

namespace thorin {

class AutoDiff : public RWPass<> {
public:
    AutoDiff(PassMan& man)
        : RWPass(man, "auto_diff")
    {}
    const Def* rewrite(const Def*) override;
};

}

#endif
