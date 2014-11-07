#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Lambda;
class Scope;

template<bool> class CFGView;
typedef CFGView<true>  F_CFG;
typedef CFGView<false> B_CFG;

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

private:
    ArrayRef<const CFGNode*> preds() const { return ArrayRef<const CFGNode*>(preds_.data(), preds_.size()); }
    ArrayRef<const CFGNode*> succs() const { return ArrayRef<const CFGNode*>(succs_.data(), succs_.size()); }

    Lambda* lambda_;
    std::vector<CFGNode*> preds_;
    std::vector<CFGNode*> succs_;
    std::vector<CFGNode*> reduced_preds_;
    std::vector<CFGNode*> reduced_succs_;

    friend class CFG;
};

class CFG {
public:
    enum class Color : uint8_t { White, Gray, Black };

    CFG(const CFG&) = delete;
    CFG& operator= (CFG) = delete;

    explicit CFG(const Scope& scope);

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes_.size(); }
    size_t sid(Lambda* lambda) const;
    size_t sid(const CFGNode* n) const { return sid(n->lambda()); }
    ArrayRef<const CFGNode*> nodes() const { return ArrayRef<const CFGNode*>(nodes_.data(), nodes_.size()); }
    ArrayRef<const CFGNode*> preds(Lambda* lambda) const { return nodes_[sid(lambda)]->preds(); }
    ArrayRef<const CFGNode*> succs(Lambda* lambda) const { return nodes_[sid(lambda)]->succs(); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const CFGNode* entry() const { return nodes_.front(); }
    const CFGNode* exit() const { return nodes_.back(); }
    const F_CFG* f_cfg() const;
    const B_CFG* b_cfg() const;
    const DomTree* domtree() const;
    const PostDomTree* postdomtree() const;
    const LoopTree* looptree() const;
    const CFGNode* lookup(Lambda* lambda) const { return nodes_[sid(lambda)]; }

private:
    void cfa();
    void reduced_visit(std::vector<Color>& colors, CFGNode* prev, CFGNode* cur);
    void link(CFGNode* src, CFGNode* dst) {
        assert(src->lambda()->intrinsic() != Intrinsic::Exit);
        src->succs_.push_back(dst);
        dst->preds_.push_back(src);
    }
    void reduced_link(CFGNode* src, CFGNode* dst) {
        if (src) {
            assert(src->lambda()->intrinsic() != Intrinsic::Exit);
            src->reduced_succs_.push_back(dst);
            dst->reduced_preds_.push_back(src);
        }
    }

    const Scope& scope_;
    Array<CFGNode*> nodes_;     // sorted in sid
    mutable AutoPtr<const F_CFG> f_cfg_;
    mutable AutoPtr<const B_CFG> b_cfg_;
    mutable AutoPtr<const LoopTree> looptree_;
};

template<bool forward = true>
class CFGView {
public:
    CFGView(const CFGView&) = delete;
    CFGView& operator= (CFGView) = delete;

    explicit CFGView(const CFG& cfg);

    const CFG& cfg() const { return cfg_; }
    size_t size() const { return rpo_.size(); }
    ArrayRef<const CFGNode*> preds(const CFGNode* n) const { return forward ? cfg().preds(n->lambda()) : cfg().succs(n->lambda()); }
    ArrayRef<const CFGNode*> succs(const CFGNode* n) const { return forward ? cfg().succs(n->lambda()) : cfg().preds(n->lambda()); }
    size_t num_preds(const CFGNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFGNode* n) const { return succs(n).size(); }
    const CFGNode* entry() const { return forward ? cfg().entry() : cfg().exit();  }
    const CFGNode* exit()  const { return forward ? cfg().exit()  : cfg().entry(); }
    size_t rpo_id(const CFGNode* n) const { return rpo_ids_[cfg().sid(n->lambda())]; }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<const CFGNode*> rpo() const { return rpo_; }
    const CFGNode* rpo(size_t i) const { return rpo_[i]; }
    /// Like \p rpo() but without \p entry()
    ArrayRef<const CFGNode*> body() const { return rpo().slice_from_begin(1); }
    const CFGNode* lookup(Lambda* lambda) const { return cfg().lookup(lambda); }
    const DomTreeBase<forward>* domtree() const;

    typedef ArrayRef<const CFGNode*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

private:
    size_t& _rpo_id(const CFGNode* n) { return rpo_ids_[cfg().sid(n->lambda())]; }
    size_t number(const CFGNode*, size_t);

    const CFG& cfg_;
    Array<size_t> rpo_ids_;     // sorted in sid
    Array<const CFGNode*> rpo_; // sorted in rpo_id
    mutable AutoPtr<const DomTreeBase<forward>> domtree_;
};

}

#endif
