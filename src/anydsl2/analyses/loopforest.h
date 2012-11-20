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

    LoopForestNode(LoopForestNode* parent, int depth)
        : parent_(parent)
        , depth_(depth)
    {
        if (parent_)
            parent_->children_.push_back(this);
    }

    int depth() const { return depth_; }
    const LoopForestNode* parent() const { return parent_; }
    ArrayRef<Lambda*> headers() const { return headers_; }
    ArrayRef<LoopForestNode*> children() const { return children_; }
    bool is_loop() const { assert(!is_root()); return !children().empty(); }
    bool is_root() const { return !parent_; }
    size_t num_headers() const { return headers().size(); }
    size_t num_children() const { return children().size(); }

private:

    LoopForestNode* parent_;
    int depth_;
    std::vector<Lambda*> headers_;
    AutoVector<LoopForestNode*> children_;

    friend class LFBuilder;
};

LoopForestNode* create_loop_forest(const Scope& scope);

std::ostream& operator << (std::ostream& o, const LoopForestNode* node);

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
