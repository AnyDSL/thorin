#ifndef THORIN_TRANSFORM_PARTIAL_EVALUATION_H
#define THORIN_TRANSFORM_PARTIAL_EVALUATION_H

#include "thorin/transform/mangle.h"

namespace thorin {

class World;

class BetaReducer : Rewriter {
public:
    BetaReducer(World& w) : Rewriter(w) {}

    void provide_arg(const Param* param, const Def* arg) {
        insert(param, arg);
    }

    const Def* reduce(const Def* odef) {
        return instantiate(odef);
    }

protected:
    const Def* rewrite(const Def* odef) override;
};

bool partial_evaluation(World&, bool lower2cff = false);

}

#endif
