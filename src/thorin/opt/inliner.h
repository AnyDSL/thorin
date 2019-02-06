#ifndef THORIN_OPT_INLINER_H
#define THORIN_OPT_INLINER_H

#include "thorin/opt/optimizer.h"

namespace thorin {

class Inliner : public Optimization {
public:
    Inliner(Optimizer& optimizer)
        : Optimization(optimizer, "Inliner")
    {}

    void enter(Lam*) override;
    const Def* visit(const Def*) override;

    size_t& uses(Lam* lam) {
        auto&& p = uses_.emplace(lam, 0);
        return p.first->second;
    }

private:
    LamMap<size_t> uses_;
};

}

#endif
