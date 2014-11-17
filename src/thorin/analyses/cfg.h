#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class CFA;
class Lambda;
class Scope;

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

template<bool> class DomTreeBase;
typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;

class LoopTree;

class CFNode {
public:
    CFNode(Lambda* lambda)
        : lambda_(lambda)
    {}

    Lambda* lambda() const { return lambda_; }

private:
    ArrayRef<const CFNode*> preds() const { return ArrayRef<const CFNode*>(preds_.data(), preds_.size()); }
    ArrayRef<const CFNode*> succs() const { return ArrayRef<const CFNode*>(succs_.data(), succs_.size()); }
    void link(CFNode* other) {
        assert(this->lambda()->intrinsic() != Intrinsic::EndScope);
        this->succs_.push_back(other);
        other->preds_.push_back(this);
    }

    Lambda* lambda_;
    std::vector<CFNode*> preds_;
    std::vector<CFNode*> succs_;

    friend class CFABuilder;
    friend class CFA;
};

class CFA {
public:
    CFA(const CFA&) = delete;
    CFA& operator= (CFA) = delete;

    explicit CFA(const Scope& scope);

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes_.size(); }
    size_t sid(Lambda* lambda) const;
    size_t sid(const CFNode* n) const { return sid(n->lambda()); }
    ArrayRef<const CFNode*> nodes() const { return ArrayRef<const CFNode*>(nodes_.data(), nodes_.size()); }
    ArrayRef<const CFNode*> preds(Lambda* lambda) const { return nodes_[sid(lambda)]->preds(); }
    ArrayRef<const CFNode*> succs(Lambda* lambda) const { return nodes_[sid(lambda)]->succs(); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const CFNode* entry() const { return nodes_.front(); }
    const CFNode* exit() const { return nodes_.back(); }
    const F_CFG* f_cfg() const;
    const B_CFG* b_cfg() const;
    const DomTree* domtree() const;
    const PostDomTree* postdomtree() const;
    const LoopTree* looptree() const;
    const CFNode* lookup(Lambda* lambda) const { auto i = sid(lambda); return i == size_t(-1) ? nullptr : nodes_[i]; }

private:
    CFNode* _lookup(Lambda* lambda) const { return nodes_[sid(lambda)]; }
    const Scope& scope_;
    Array<CFNode*> nodes_;     // sorted in sid
    mutable AutoPtr<const F_CFG> f_cfg_;
    mutable AutoPtr<const B_CFG> b_cfg_;
    mutable AutoPtr<const LoopTree> looptree_;

    friend class CFABuilder;
};

template<bool forward = true>
class CFG {
public:
    CFG(const CFG&) = delete;
    CFG& operator= (CFG) = delete;

    explicit CFG(const CFA&);

    const CFA& cfa() const { return cfa_; }
    size_t size() const { return rpo_.size(); }
    ArrayRef<const CFNode*> preds(const CFNode* n) const { return forward ? cfa().preds(n->lambda()) : cfa().succs(n->lambda()); }
    ArrayRef<const CFNode*> succs(const CFNode* n) const { return forward ? cfa().succs(n->lambda()) : cfa().preds(n->lambda()); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    const CFNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const CFNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }
    size_t rpo_id(const CFNode* n) const { return rpo_ids_[cfa().sid(n->lambda())]; }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<const CFNode*> rpo() const { return rpo_; }
    const CFNode* rpo(size_t i) const { return rpo_[i]; }
    /// Like \p rpo() but without \p entry()
    ArrayRef<const CFNode*> body() const { return rpo().slice_from_begin(1); }
    const CFNode* lookup(Lambda* lambda) const { return cfa().lookup(lambda); }
    const DomTreeBase<forward>* domtree() const;

    typedef ArrayRef<const CFNode*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

private:
    size_t& _rpo_id(const CFNode* n) { return rpo_ids_[cfa().sid(n->lambda())]; }
    size_t number(const CFNode*, size_t);

    const CFA& cfa_;
    Array<size_t> rpo_ids_;     // sorted in sid
    Array<const CFNode*> rpo_; // sorted in rpo_id
    mutable AutoPtr<const DomTreeBase<forward>> domtree_;
};

}

#endif
