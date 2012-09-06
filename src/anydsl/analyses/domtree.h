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

const DomNode* calc_domtree(const Lambda* entry, const LambdaSet& scope);
const DomNode* calc_domtree(const Lambda* entry);

} // namespace anydsl

#endif
