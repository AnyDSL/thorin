#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Lambda;
class Scope;

class CFGNode {
public:
    CFGNode(Lambda* lambda)
        : lambda_(lambda)
    {}

    Lambda* lambda() const { return lambda_; }
    bool is_entry() const { return preds_.empty(); }
    bool is_exit() const { return succs_.empty(); }
    const std::vector<CFGNode*> preds() const { return preds_; }
    const std::vector<CFGNode*> succs() const { return succs_; }

private:
    Lambda* lambda_;
    std::vector<CFGNode*> preds_;
    std::vector<CFGNode*> succs_;
};

class CFG {
public:
    CFG(const Scope& scope);

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes_.size(); }
    bool empty() const { return size() == 0; }
    ArrayRef<const CFGNode*> nodes() const { return ArrayRef<const CFGNode*>(nodes_.data(), nodes_.size()); }

private:
    void cfa();

    const Scope& scope_;
    AutoVector<CFGNode*> nodes_;
};

}

#endif
