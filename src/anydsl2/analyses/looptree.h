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
 * A \p LoopTreeNode consists of a set of header \p Lambda%s and \p LoopTreeNode%s as children.
 * The root node is a \p LoopTreeNode without any headers but further \p LoopTreeNode children.
 * Thus, the forest is pooled into a tree.
 */
class LoopTreeNode {
public:

    LoopTreeNode(LoopTreeNode* parent, int depth)
        : parent_(parent)
        , depth_(depth)
    {
        if (parent_)
            parent_->children_.push_back(this);
    }

    int depth() const { return depth_; }
    const LoopTreeNode* parent() const { return parent_; }
    ArrayRef<Lambda*> headers() const { return headers_; }
    ArrayRef<LoopTreeNode*> children() const { return children_; }
    const LoopTreeNode* child(size_t i) const { return children_[i]; }
    bool is_root() const { return !parent_; }
    size_t num_headers() const { return headers().size(); }
    size_t num_children() const { return children().size(); }

private:

    LoopTreeNode* parent_;
    int depth_;
    std::vector<Lambda*> headers_;
    AutoVector<LoopTreeNode*> children_;
    std::vector<Edge> backedges_;
    std::vector<Edge> entries_;

    friend class LFBuilder;
};

//class LoopTree {
//public:

    //LoopTree(Scope&);

    //Array<LoopTreeNode*> nodes_;

    //DomNode* lookup(Lambda* lambda) { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    //const DomNode* lookup(Lambda* lambda) const { return const_cast<DomTreeBase*>(this)->lookup(lambda); }
//};

/**
 * Calculates a loop nesting forest rooted at the returned \p LoopTreeNode.
 * You will manually have to delete this returned node in order to free memory again.
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
LoopTreeNode* create_loop_forest(const Scope& scope);

std::ostream& operator << (std::ostream& o, const LoopTreeNode* node);

class LoopInfo {
public:

    LoopInfo(const Scope& scope)
        : scope_(scope)
        , depth_(scope.size())
    {
        build_infos();
    }

    const Scope& scope() const { return scope_; }
    int depth(size_t sid) const { return depth_[sid]; }
    int depth(Lambda* lambda) const { 
        assert(scope_.contains(lambda)); 
        return depth(lambda->sid()); 
    }

private:

    void build_infos();
    void visit(const LoopTreeNode* n);

    const Scope& scope_;
    Array<int> depth_;
};

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
