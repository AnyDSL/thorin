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

    LoopForestNode(LoopForestNode* parent)
        : parent_(parent)
        , depth_(0)
    {
        if (parent_)
            parent_->children_.push_back(this);
    }

    int depth() const { return depth_; }
    ArrayRef<Lambda*> headers() const { return headers_; }
    ArrayRef<LoopForestNode*> children() const { return children_; }
    LoopForestNode* parent() const { return parent_; }

private:

    LoopForestNode* parent_;
    int depth_;
    std::vector<Lambda*> headers_;
    AutoVector<LoopForestNode*> children_;

    friend class LFBuilder;
};

LoopForestNode* create_loop_forest(const Scope& scope);

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
