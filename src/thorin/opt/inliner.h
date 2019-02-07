#ifndef THORIN_OPT_INLINER_H
#define THORIN_OPT_INLINER_H

#include "thorin/opt/optimizer.h"

namespace thorin {

class Inliner : public Optimization {
public:
    Inliner(Optimizer& optimizer)
        : Optimization(optimizer)
    {}

    const Def* rewrite(const Def*) override;
    void analyze(const Def*) override;

private:
    size_t& uses(Lam* lam) { return uses_.emplace(lam, 0).first->second; }

    LamMap<size_t> uses_;
};

}

#endif
