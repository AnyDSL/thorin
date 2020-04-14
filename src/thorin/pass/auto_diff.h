#ifndef THORIN_PASS_AUTO_DIFF_H
#define THORIN_PASS_AUTO_DIFF_H

#include "pass.h"

namespace thorin {

class AutoDiff : public PassBase {
public:
    AutoDiff(PassMan& man, size_t idx)
        : PassBase(man, idx) {}
    const Def* rewrite(const Def*) override;
};

} // namespace thorin

#endif // THORIN_PASS_AUTO_DIFF_H
