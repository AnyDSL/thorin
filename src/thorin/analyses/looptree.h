#ifndef THORIN_ANALYSES_LOOPTREE_H
#define THORIN_ANALYSES_LOOPTREE_H

#include <vector>

#include "thorin/analyses/cfg.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"

namespace thorin {

class LoopHeader;

/**
 * @brief Represents a node of a loop nesting forest.
 *
 * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
 * for an introduction to loop nesting forests.
 * A @p LoopNode consists of a set of header @p CFNode%s.
 * The header CFNode%s are the set of CFNode%s not dominated by any other @p CFNode within the loop.
 * The root node is a @p LoopHeader without any CFNode%s but further @p LoopNode children and @p depth_ -1.
 * Thus, the forest is pooled into a tree.
 */
class LoopNode : public MagicCast<LoopNode> {
public:
    LoopNode(LoopHeader* parent, int depth, const std::vector<const CFNode*>&);

    int depth() const { return depth_; }
    const LoopHeader* parent() const { return parent_; }
    ArrayRef<const CFNode*> cf_nodes() const { return cf_nodes_; }
    size_t num_cf_nodes() const { return cf_nodes().size(); }
    virtual void dump() const = 0;

protected:
    std::ostream& indent() const;

    LoopHeader* parent_;
    int depth_;
    std::vector<const CFNode*> cf_nodes_;
};

/// A LoopHeader owns further @p LoopNode%s as children.
class LoopHeader : public LoopNode {
public:
    LoopHeader(LoopHeader* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
        : LoopNode(parent, depth, cf_nodes)
    {}

    ArrayRef<LoopNode*> children() const { return children_; }
    const LoopNode* child(size_t i) const { return children_[i]; }
    size_t num_children() const { return children().size(); }
    bool is_root() const { return parent_ == 0; }
    virtual void dump() const;

private:
    AutoVector<LoopNode*> children_;

    friend class LoopNode;
    friend class LoopTreeBuilder;
};

class LoopLeaf : public LoopNode {
public:
    explicit LoopLeaf(size_t dfs_id, LoopHeader* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
        : LoopNode(parent, depth, cf_nodes)
        , dfs_id_(dfs_id)
    {
        assert(num_cf_nodes() == 1);
    }

    const CFNode* cf_node() const { return cf_nodes().front(); }
    size_t dfs_id() const { return dfs_id_; }
    virtual void dump() const;

private:
    size_t dfs_id_;
};

/**
 * @brief Calculates a loop nesting forest rooted at @p root_.
 * 
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
class LoopTree {
public:
    LoopTree(const LoopTree&) = delete;
    LoopTree& operator= (LoopTree) = delete;

    explicit LoopTree(const F_CFG& cfg);

    const F_CFG& cfg() const { return cfg_; }
    const LoopHeader* root() const { return root_; }
    int depth(const CFNode* n) const { return cf2leaf(n)->depth(); }
    size_t cf2dfs(const CFNode* n) const { return cf2leaf(n)->dfs_id(); }
    void dump() const { root()->dump(); }
    const LoopLeaf* cf2leaf(const CFNode* n) const { return find(cf2leaf_, n); }
    const LoopHeader* cf2header(const CFNode*) const;

private:
    const F_CFG& cfg_;
    F_CFG::Map<LoopLeaf*> cf2leaf_;
    AutoPtr<LoopHeader> root_;

    friend class LoopTreeBuilder;
};

}

#endif
