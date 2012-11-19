#ifndef ANALYSES_LOOPS_H
#define ANALYSES_LOOPS_H

#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

#include <vector>

namespace anydsl2 {

class Lambda;
class Scope;
class World;

class LoopForestNode {
public:

    LoopForestNode(int depth, ArrayRef<Lambda*> lambdas)
        : depth_(depth)
        , lambdas_(lambdas)
    {}

    int depth() const { return depth_; }
    ArrayRef<Lambda*> lambdas() const { return lambdas_; }
    ArrayRef<LoopForestNode*> children() const { return children_; }

private:

    int depth_;
    Array<Lambda*> lambdas_;
    AutoVector<LoopForestNode*> children_;

    friend class LFBuilder;
};

LoopForestNode* create_loop_forest(const Scope& scope);

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
