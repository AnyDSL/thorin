#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

class Def;
class Lambda;
class Scope;
class World;

class DomNode {
public:
    explicit DomNode(Lambda* lambda)
        : lambda_(lambda)
        , idom_(nullptr)
    {}

    Lambda* lambda() const { return lambda_; }
    const DomNode* idom() const { return idom_; }
    const std::vector<const DomNode*>& children() const { return children_; }
    size_t num_children() const { return children_.size(); }
    bool entry() const { return idom_ == this; }
    int depth() const;
    void dump() const;

private:
    Lambda* lambda_;
    DomNode* idom_;
    AutoVector<const DomNode*> children_;

    friend class DomTree;
};

class DomTree {
public:
    explicit DomTree(const Scope& scope, bool is_forward = true);

    const ScopeView& scope_view() const { return scope_view_; }
    const DomNode* root() const { return root_; }
    int depth(Lambda* lambda) const { return lookup(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const {
        return const_cast<DomTree*>(this)->lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j));
    }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }
    const DomNode* lookup(Lambda* lambda) const { return find(map_, lambda); }
    void dump() const { root()->dump(); }

private:
    DomNode* lookup(Lambda* lambda) { return map_[lambda]; }
    void create();
    DomNode* lca(DomNode* i, DomNode* j);

    ScopeView scope_view_;
    AutoPtr<DomNode> root_;
    LambdaMap<DomNode*> map_;
};

}

#endif
