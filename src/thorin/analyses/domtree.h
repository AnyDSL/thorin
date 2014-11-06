#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

class DomNode {
public:
    explicit DomNode(Lambda* lambda)
        : lambda_(lambda)
    {}

    Lambda* lambda() const { return lambda_; }
    const DomNode* idom() const { return idom_; }
    const std::vector<const DomNode*>& children() const { return children_; }
    size_t num_children() const { return children_.size(); }
    bool entry() const { return idom_ == this; }
    int depth() const { return depth_; }
    size_t max_sid() const;
    void dump() const;

private:
    Lambda* lambda_;
    DomNode* idom_ = nullptr;
    AutoVector<const DomNode*> children_;
    int depth_;
    size_t max_sid_;

    template<bool> friend class DomTreeBase;
};

template<bool forward>
class DomTreeBase {
public:
    explicit DomTreeBase(const Scope& scope);

    const ScopeView<forward>& scope_view() const { return scope_view_; }
    size_t sid(Lambda* lambda) const { return scope_view().sid(lambda); };
    size_t sid(DomNode* n) const { return sid(n->lambda()); }
    const DomNode* root() const { return root_; }
    int depth(Lambda* lambda) const { return lookup(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const {
        return const_cast<DomTreeBase*>(this)->lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j));
    }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }
    const DomNode* lookup(Lambda* lambda) const { return lookup(sid(lambda)); }
    const DomNode* lookup(size_t sid) const { return nodes_[sid]; }
    void dump() const { root()->dump(); }

private:
    DomNode*& lookup(Lambda* lambda) { return nodes_[sid(lambda)]; }
    void create();
    size_t postprocess(DomNode* n, int depth);
    DomNode* lca(DomNode*, DomNode*);

    ScopeView<forward> scope_view_;
    AutoPtr<DomNode> root_;
    std::vector<DomNode*> nodes_;
};

}

#endif
