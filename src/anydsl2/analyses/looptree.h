#ifndef ANALYSES_LOOPS_H
#define ANALYSES_LOOPS_H

#include <vector>

#include "anydsl2/analyses/scope_analysis.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class Lambda;
class Scope;
class World;

struct Edge {
    Edge() {}
    Edge(Lambda* src, Lambda* dst) 
        : src_(src)
        , dst_(dst)
    {}

    Lambda* src() const { return src_; }
    Lambda* dst() const { return dst_; }

private:

    Lambda* src_;
    Lambda* dst_;
};

/**
 * Represents a node of a loop nesting forest.
 * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
 * for an introduction to loop nesting forests.
 * A \p LoopNode consists of a set of header \p Lambda%s and \p LoopNode%s as children.
 * The root node is a \p LoopNode without any headers but further \p LoopNode children.
 * Thus, the forest is pooled into a tree.
 */
class LoopNode {
public:

    LoopNode(LoopNode* parent, int depth)
        : parent_(parent)
        , depth_(depth)
    {
        if (parent_)
            parent_->children_.push_back(this);
    }

    int depth() const { return depth_; }
    const LoopNode* parent() const { return parent_; }
    ArrayRef<Lambda*> headers() const { return headers_; }
    ArrayRef<LoopNode*> children() const { return children_; }
    const LoopNode* child(size_t i) const { return children_[i]; }
    bool is_root() const { return !parent_; }
    size_t num_headers() const { return headers().size(); }
    size_t num_children() const { return children().size(); }
    bool is_leaf() const { assert(num_headers() == 1); return num_children() == 0; }
    const std::vector<Edge>& backedges() const { assert(!is_leaf()); return backedges_or_exits_; }
    const std::vector<Edge>& exits() const { assert(is_leaf()); return backedges_or_exits_; }
    Lambda* lambda() const { assert(is_leaf()); return headers().front(); }

private:

    LoopNode* parent_;
    int depth_;
    std::vector<Lambda*> headers_;
    AutoVector<LoopNode*> children_;
    std::vector<Edge> backedges_or_exits_;
    std::vector<Edge> entries_;

    friend class LoopTreeBuilder;
};

/**
 * Calculates a loop nesting forest rooted at \p root_.
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
class LoopTree : public ScopeAnalysis<LoopNode, true, false /*do not auto-destroy nodes*/> {
public:

    typedef ScopeAnalysis<LoopNode, true, false> Super;

    explicit LoopTree(const Scope& scope)
        : Super(scope)
    {
        create();
    }

    const LoopNode* root() const { return root_; }
    int depth(Lambda* lambda) const { return Super::lookup(lambda)->depth(); }

private:

    AutoPtr<LoopNode> root_;

    void create();

    friend class LoopTreeBuilder;
};

LoopNode* create_loop_forest(const Scope& scope);

std::ostream& operator << (std::ostream& o, const LoopNode* node);

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
