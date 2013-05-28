#ifndef ANYDSL2_ANALYSES_DOMTREE_H
#define ANYDSL2_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class DomNode;
class Def;
class Lambda;
class Scope;
class World;

typedef std::vector<const DomNode*> DomNodes;

class DomNode {
public:

    DomNode(Lambda* lambda);

    Lambda* lambda() const { return lambda_; }
    /// Returns post-order number of lambda in scope.
    const DomNode* idom() const { return idom_; }
    const DomNodes& children() const { return children_; }
    bool entry() const { return idom_ == this; }
    int depth() const;

private:

    Lambda* lambda_;
    DomNode* idom_;
    DomNodes children_;

    friend class DomTree;
};

class DomTree {
public:

    DomTree(const Scope& scope, bool is_postdomtree);
    ~DomTree();

    const Scope& scope() const { return scope_; }
    bool is_postdomtree() const { return is_postdomtree_; }
    ArrayRef<const DomNode*> nodes() const { return ArrayRef<const DomNode*>(nodes_.begin(), nodes_.size()); }
    size_t size() const;
    const DomNode* node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    int depth(Lambda* lambda) const { return node(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const { return lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j)); }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }
    size_t index(DomNode* n) const { return index(n->lambda()); }
    /// Returns \p lambda%'s \p backwards_sid() in the case this a postdomtree 
    /// or \p lambda%'s sid() if this is an ordinary domtree.
    size_t index(Lambda* lambda) const { return is_postdomtree_ ? lambda->backwards_sid() : lambda->sid(); }

private:

    DomNode* lca(DomNode* i, DomNode* j);
    template<bool post> void create();
    DomNode* lookup(Lambda* lambda) { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    const DomNode* lookup(Lambda* lambda) const { return const_cast<DomTree*>(this)->lookup(lambda); }

    const Scope& scope_;
    Array<DomNode*> nodes_;
    bool is_postdomtree_;
};

} // namespace anydsl2

#endif
