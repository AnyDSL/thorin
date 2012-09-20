#ifndef ANYDSL_ANALYSES_DOMTREE_H
#define ANYDSL_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

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
    ~DomNode();

    const Lambda* lambda() const { return lambda_; }
    /// Returns post-order number of lambda in scope.
    size_t index() const { return index_; }
    const DomNode* idom() const { return idom_; }
    const DomNodes& children() const { return children_; }
    bool entry() const { return idom_ == this; }
    int depth() const;

private:

    const Lambda* lambda_;
    DomNode* idom_;
    size_t index_;
    DomNodes children_;

    friend class DomBuilder;
};

class DomTree {
public:

    DomTree(size_t size, const DomNode* root);
    DomTree(const DomTree& tree) : size_(0), root_(0), bfs_(0) { ANYDSL_UNREACHABLE; }
    ~DomTree() { delete root_; }

    const DomNode* root() const { return root_; }
    size_t size() const { return size_; }
    ArrayRef<const DomNode*> bfs() const { return bfs_; }
    const DomNode* bfs(size_t i) const { return bfs_[i]; }

    bool dominates(const DomNode* a, const DomNode* b);
    bool strictly_dominates(const DomNode* a, const DomNode* b) { return a != b && dominates(a, b); }

private:

    size_t size_;
    const DomNode* root_;
    Array<const DomNode*> bfs_;
};

DomTree calc_domtree(const Scope& scope);

} // namespace anydsl

#endif
