#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/util/array.h"

namespace thorin {

class Lambda;
class Scope;

class CFGNode {
public:
    CFGNode(Lambda* lambda)
        : lambda_(lambda)
    {}

    Lambda* lambda() const { return lambda_; }

private:
    Lambda* lambda_;
    std::vector<CFGNode*> preds_;
    std::vector<CFGNode*> sucss_;
};

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
