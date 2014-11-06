#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Lambda;
class Scope;

template<bool> class DomTreeBase;
typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;
class LoopTree;

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
    const DomTree* domtree() const;
    const PostDomTree* postdomtree() const;
    const LoopTree* looptree() const;

private:
    template<class T> T* lazy(AutoPtr<T>& ptr) const { return ptr ? ptr : ptr = new T(*this); }

    void link(CFGNode* src, CFGNode* dst) {
        src->succs_.push_back(dst);
        dst->preds_.push_back(src);
    }
    void cfa();

    const Scope& scope_;
    Array<CFGNode*> nodes_;
    mutable AutoPtr<const DomTree> domtree_;
    mutable AutoPtr<const PostDomTree> postdomtree_;
    mutable AutoPtr<const LoopTree> looptree_;
};

template<bool forward = true>
class CFGView {
public:
    CFGView(const CFG& cfg)
        : cfg_(cfg)
    {}

    const CFG& cfg() const { return cfg_; }
    size_t size() const { return rpo_.size(); }
    ArrayRef<const CFGNode*> preds(const CFGNode* n) const { return forward ? cfg().preds(n->lambda()) : cfg().succs(n->lambda()); }
    ArrayRef<const CFGNode*> succs(const CFGNode* n) const { return forward ? cfg().succs(n->lambda()) : cfg().preds(n->lambda()); }
    size_t num_preds(const CFGNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFGNode* n) const { return succs(n).size(); }
    const CFGNode* entry() const { return rpo().front(); }
    const CFGNode* exit()  const { return rpo().end(); }
    size_t rpo_id(const CFGNode* n) const { return rpo_ids_[cfg().sid(n->lambda())]; }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<const CFGNode*> rpo() const { return rpo_; }
    const CFGNode* rpo(size_t i) const { return rpo_[i]; }
    /// Like \p rpo() but without \p entry()
    ArrayRef<const CFGNode*> body() const { return rpo().slice_from_begin(1); }

    typedef ArrayRef<const CFGNode*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

private:
    const CFG& cfg_;
    Array<size_t> rpo_ids_;
    Array<const CFGNode*> rpo_;
};

}

#endif
