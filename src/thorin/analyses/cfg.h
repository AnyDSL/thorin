#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/util/array.h"

namespace thorin {

class Lambda;
class Scope;

class CFG {
public:
    CFG(const Scope& scope);

    const Scope& scope() const { return scope_; }
    void cfa();

private:
    const Scope& scope_;
    Array<Lambda*> lambdas_;
};

}

#endif
