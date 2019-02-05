#ifndef THORIN_OPT_INLINER_H
#define THORIN_OPT_INLINER_H

#include "thorin/opt/optimizer.h"

namespace thorin {

class Inliner : public Optimization {
public:
    Inliner()
        : Optimization("Inliner")
    {}

    void visit(Lam*) override;
    void visit(const Def*) override;
};

}

#endif
