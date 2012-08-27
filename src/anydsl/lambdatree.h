#ifndef ANYDSL_LAMBDA_TREE_H
#define ANYDSL_LAMBDA_TREE_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class Lambda;
class World;

class LambdaNode;
typedef boost::unordered_set<const LambdaNode*> NodeSet;

class LambdaNode {
public:

    ~LambdaNode();

    const Lambda* lambda() const { return lambda_; }
    const NodeSet& children() const { return children_; }
    const LambdaNode* idom() const { return idom_; }

private:

    const Lambda* lambda_;
    const LambdaNode* idom_;
    NodeSet children_;
};

const LambdaNode* build_lambda_tree(const World& world);

} // namespace anydsl

#endif
