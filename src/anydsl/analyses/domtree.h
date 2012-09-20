#ifndef ANYDSL_ANALYSES_DOMTREE_H
#define ANYDSL_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

#include "anydsl/lambda.h"
#include "anydsl/util/array.h"

namespace anydsl {

class DomNode;
class Def;
class Lambda;
class Scope;
class World;

typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef std::vector<const DomNode*> DomNodes;

class DomNode {
public:

    DomNode(const Lambda* lambda);

    const Lambda* lambda() const { return lambda_; }
    /// Returns post-order number of lambda in scope.
    const DomNode* idom() const { return idom_; }
    const DomNodes& children() const { return children_; }
    bool entry() const { return idom_ == this; }
    int depth() const;
    size_t sid() const { return lambda()->sid(); }


private:

    const Lambda* lambda_;
    DomNode* idom_;
    DomNodes children_;

    friend class DomTree;
};

class DomTree {
public:

    explicit DomTree(const Scope& scope)
        : scope_(scope)
        , bfs_(size())
        , nodes_(size())
    {
        create();
    }
    ~DomTree();

    const Scope& scope() const { return scope_; }
    const DomNode* entry() const;
    size_t size() const;
    ArrayRef<const DomNode*> bfs() const { return bfs_; }
    ArrayRef<const DomNode*> nodes() const { return ArrayRef<const DomNode*>(nodes_.begin(), nodes_.size()); }
    const DomNode* node(size_t sid) const { return nodes_[sid]; }
    const DomNode* node_(const Lambda* lambda) const { return nodes_[lambda->sid()]; }
    const DomNode* bfs(size_t i) const { return bfs_[i]; }

    bool dominates(const DomNode* a, const DomNode* b);
    bool strictly_dominates(const DomNode* a, const DomNode* b) { return a != b && dominates(a, b); }

private:

    void create();
    DomNode* node(const Lambda* lambda) { return nodes_[lambda->sid()]; }
    DomNode* intersect(DomNode* i, DomNode* j);

    const Scope& scope_;
    Array<const DomNode*> bfs_;
    Array<DomNode*> nodes_;
};

} // namespace anydsl

#endif
