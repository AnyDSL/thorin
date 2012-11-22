#ifndef ANALYSES_LOOPS_H
#define ANALYSES_LOOPS_H

#include <memory>
#include <vector>

#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

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

class LoopInfo {
public:

    LoopInfo(const Scope& scope)
        : scope_(scope)
        , depth_(scope.size())
        , root_(create_loop_forest(scope))
    {
        build_infos();
    }

    const Scope& scope() const { return scope_; }
    LoopForestNode* root() const { return root_.get(); }
    int depth(size_t sid) const { return depth_[sid]; }
    int depth(Lambda* lambda) const { 
        assert(scope_.contains(lambda)); 
        return depth(lambda->sid()); 
    }

private:

    void build_infos();
    void visit(LoopForestNode* n);

    const Scope& scope_;
    Array<int> depth_;
    std::auto_ptr<LoopForestNode> root_;
};

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
