#ifndef ANYDSL_ANALYSES_DOMTREE_H
#define ANYDSL_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class DomNode;
class Lambda;
class World;

typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef boost::unordered_set<const DomNode*> DomNodes;

class DomNode {
public:

    DomNode(const Lambda* lambda) : lambda_(lambda) {}

    const Lambda* lambda() const { return lambda_; }
    const DomNode* idom() const { return idom_; }
    const DomNodes& children() const { return children_; }

private:

    const Lambda* lambda_;
    DomNode* idom_;
    DomNodes children_;
};

const DomNode* calc_domtree(const Lambda* entry, const LambdaSet& scope);
const DomNode* calc_domtree(const Lambda* entry);
void calc_domtree(const World& world);

} // namespace anydsl

#endif
