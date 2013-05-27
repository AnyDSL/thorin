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
    size_t sid() const;

private:

    Lambda* lambda_;
    DomNode* idom_;
    DomNodes children_;

    friend class DomTree;
};

class DomTree {
public:

    DomTree(const Scope& scope, bool post);
    ~DomTree();

    const Scope& scope() const { return scope_; }
    size_t size() const;
    ArrayRef<const DomNode*> nodes() const { return ArrayRef<const DomNode*>(nodes_.begin(), nodes_.size()); }
    const DomNode* node(size_t sid) const { return nodes_[sid]; }
    const DomNode* node(Lambda* lambda) const;
    int depth(Lambda* lambda) const { return node(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    static const DomNode* lca(const DomNode* i, const DomNode* j) { 
        return lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j)); 
    }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }

private:

    static DomNode* lca(DomNode* i, DomNode* j);
    static DomNode* post_lca(DomNode* i, DomNode* j);
    template<bool post> void create();
    void create_postdom();
    DomNode* lookup(Lambda* lambda) const;

    const Scope& scope_;
    Array<DomNode*> nodes_;
};

} // namespace anydsl2

#endif
