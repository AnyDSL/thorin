#ifndef ANYDSL_LAMBDA_TREE_H
#define ANYDSL_LAMBDA_TREE_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class Def;
class Lambda;
class World;

class LambdaNode;
typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef boost::unordered_set<const LambdaNode*> LambdaNodes;

class LambdaNode {
public:

    LambdaNode(const Lambda* lambda);
    ~LambdaNode();

    const Lambda* lambda() const { return lambda_; }
    const LambdaNodes& children() const { return children_; }
    LambdaNode* parent() const { return parent_; }
    bool top() const { return parent() == this; }

public:

    const Lambda* lambda_;
    LambdaNode* parent_;
    LambdaNodes children_;
};

void build_lambda_forest(const World& world);
void build_lambda_forest(const LambdaSet& lambdas);

} // namespace anydsl

#endif
