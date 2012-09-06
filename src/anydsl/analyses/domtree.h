#ifndef ANYDSL_ANALYSES_DOMTREE_H
#define ANYDSL_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

#include "anydsl/util/array.h"

namespace anydsl {

class DomNode;
class Lambda;
class World;

typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef boost::unordered_set<const DomNode*> DomNodes;

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

private:

    const Lambda* lambda_;
    DomNode* idom_;
    size_t index_;
    DomNodes children_;

    friend class DomBuilder;
};

class DomTree {
public:

    DomTree(size_t size, const DomNode* root)
        : size_(size)
        , root_(root)
    {}

    const DomNode* root() const { return root_; }
    size_t size() const { return size_; }

private:

    size_t size_;
    const DomNode* root_;
};

DomTree calc_domtree(const Lambda* entry, const LambdaSet& scope);
DomTree calc_domtree(const Lambda* entry);

} // namespace anydsl

#endif
