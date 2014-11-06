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
    ArrayRef<const CFGNode*> preds() const { return ArrayRef<const CFGNode*>(preds_.data(), preds_.size()); }
    ArrayRef<const CFGNode*> succs() const { return ArrayRef<const CFGNode*>(succs_.data(), succs_.size()); }
    bool is_entry() const { return preds_.empty(); }
    bool is_exit() const { return succs_.empty(); }

private:
    Lambda* lambda_;
    std::vector<CFGNode*> preds_;
    std::vector<CFGNode*> succs_;

    friend class CFG;
};

class CFG {
public:
    CFG(const Scope& scope);

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes_.size(); }
    bool empty() const { return size() == 0; }
    size_t sid(Lambda* lambda) const;
    ArrayRef<const CFGNode*> nodes() const { return ArrayRef<const CFGNode*>(nodes_.data(), nodes_.size()); }
    ArrayRef<const CFGNode*> preds(Lambda* lambda) const { return nodes_[sid(lambda)]->preds(); }
    ArrayRef<const CFGNode*> succs(Lambda* lambda) const { return nodes_[sid(lambda)]->succs(); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const CFGNode* entry() const { return nodes_.front(); }
    const CFGNode* exit() const { return nodes_.back(); }

private:
    void link(CFGNode* src, CFGNode* dst) {
        src->succs_.push_back(dst);
        dst->preds_.push_back(src);
    }
    void cfa();

    const Scope& scope_;
    Array<CFGNode*> nodes_;
};

template<bool forward = true>
class CFGView {
public:
    CFGView(const CFG& cfg)
        : cfg_(cfg)
    {}

    const CFG& cfg() const { return cfg_; }
    ArrayRef<const CFGNode*> preds(Lambda* lambda) const { return forward ? cfg().preds(lambda) : cfg().succs(lambda); }
    ArrayRef<const CFGNode*> succs(Lambda* lambda) const { return forward ? cfg().succs(lambda) : cfg().succs(lambda); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const CFGNode* entry() const { return forward ? cfg().entry() : cfg().exit();  }
    const CFGNode* exit()  const { return forward ? cfg().exit()  : cfg().entry(); }

private:
    const CFG& cfg_;
    Array<size_t> rpo_ids_;
    Array<const CFGNode*> rpo_;
};

}

#endif
