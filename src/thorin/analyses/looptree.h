#ifndef THORIN_ANALYSES_LOOPTREE_H
#define THORIN_ANALYSES_LOOPTREE_H

#include <vector>

#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"

namespace thorin {

class CFGNode;
template<bool> class CFGView;
typedef HashSet<const CFGNode*> CFGNodeSet;
template<class To>
using CFGNodeMap = HashMap<const CFGNode*, To>;
class LoopHeader;

/**
 * Represents a node of a loop nesting forest.
 * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
 * for an introduction to loop nesting forests.
 * A \p LoopNode consists of a set of header \p CFGNode%s.
 * The header CFGNode%s are the set of CFGNode%s not dominated by any other cfg_node within the loop.
 * The root node is a \p LoopHeader without any CFGNode%s but further \p LoopNode children and \p depth_ -1.
 * Thus, the forest is pooled into a tree.
 */
class LoopNode : public MagicCast<LoopNode> {
public:
    LoopNode(LoopHeader* parent, int depth, const std::vector<const CFGNode*>& cfg_nodes);

    int depth() const { return depth_; }
    const LoopHeader* parent() const { return parent_; }
    ArrayRef<const CFGNode*> cfg_nodes() const { return cfg_nodes_; }
    size_t num_cfg_nodes() const { return cfg_nodes().size(); }
    virtual void dump() const = 0;

protected:
    std::ostream& indent() const;

    LoopHeader* parent_;
    int depth_;
    std::vector<const CFGNode*> cfg_nodes_;
};

/// A LoopHeader owns further \p LoopNode%s as children.
class LoopHeader : public LoopNode {
public:
    struct Edge {
        Edge() {}
        Edge(const CFGNode* src, const CFGNode* dst, int levels)
            : src_(src)
            , dst_(dst)
            , levels_(levels)
        {}

        const CFGNode* src() const { return src_; }
        const CFGNode* dst() const { return dst_; }
        int levels() const { return levels_; }
        void dump();

    private:
        const CFGNode* src_;
        const CFGNode* dst_;
        int levels_;
    };

    explicit LoopHeader(LoopHeader* parent, int depth, const std::vector<const CFGNode*>& cfg_nodes)
        : LoopNode(parent, depth, cfg_nodes)
        , dfs_begin_(0)
        , dfs_end_(-1)
    {}

    ArrayRef<LoopNode*> children() const { return children_; }
    const LoopNode* child(size_t i) const { return children_[i]; }
    size_t num_children() const { return children().size(); }
    const std::vector<Edge>& entry_edges() const { return entry_edges_; }
    const std::vector<Edge>& exit_edges() const { return exit_edges_; }
    const std::vector<Edge>& back_edges() const { return back_edges_; }
    /// Set of cfg_nodes not dominated by any other cfg_node within the loop. Same as \p cfg_nodes() as \p CFGNodeSet.
    const CFGNodeSet& headers() const { return headers_; }
    /// Set of cfg_nodes dominating the loop. They are not within the loop.
    const CFGNodeSet& preheaders() const { return preheaders_; }
    /// Set of cfg_nodes which jump to one of the headers.
    const CFGNodeSet& latches() const { return latches_; }
    /// Set of cfg_nodes which jump out of the loop.
    const CFGNodeSet& exitings() const { return exitings_; }
    /// Set of cfg_nodes jumped to via exiting cfg_nodes.
    const CFGNodeSet& exits() const { return exits_; }
    bool is_root() const { return parent_ == 0; }
    size_t dfs_begin() const { return dfs_begin_; };
    size_t dfs_end() const { return dfs_end_; }
    virtual void dump() const;

private:
    size_t dfs_begin_;
    size_t dfs_end_;
    AutoVector<LoopNode*> children_;
    std::vector<Edge> entry_edges_;
    std::vector<Edge> exit_edges_;
    std::vector<Edge> back_edges_;
    CFGNodeSet headers_;
    CFGNodeSet preheaders_;
    CFGNodeSet latches_;
    CFGNodeSet exits_;
    CFGNodeSet exitings_;

    friend class LoopNode;
    friend class LoopTreeBuilder;
};

class LoopLeaf : public LoopNode {
public:
    explicit LoopLeaf(size_t dfs_id, LoopHeader* parent, int depth, const std::vector<const CFGNode*>& cfg_nodes)
        : LoopNode(parent, depth, cfg_nodes)
        , dfs_id_(dfs_id)
    {
        assert(num_cfg_nodes() == 1);
    }

    const CFGNode* cfg_node() const { return cfg_nodes().front(); }
    size_t dfs_id() const { return dfs_id_; }
    virtual void dump() const;

private:
    size_t dfs_id_;
};

/**
 * Calculates a loop nesting forest rooted at \p root_.
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
class LoopTree {
public:
    explicit LoopTree(const CFGView<true>& cfg);

    const CFGView<true>& cfg() const { return cfg_; }
    const LoopHeader* root() const { return root_; }
    int depth(const CFGNode* cfg_node) const { return cfg_node2leaf(cfg_node)->depth(); }
    size_t cfg_node2dfs(const CFGNode* cfg_node) const { return cfg_node2leaf(cfg_node)->dfs_id(); }
    bool contains(const LoopHeader* header, const CFGNode* cfg_node) const;
    ArrayRef<const LoopLeaf*> loop(const LoopHeader* header) {
        return ArrayRef<const LoopLeaf*>(dfs_leaves_.data() + header->dfs_begin(), header->dfs_end() - header->dfs_begin());
    }
    Array<const CFGNode*> loop_cfg_nodes(const LoopHeader* header);
    Array<const CFGNode*> loop_cfg_nodes_in_rpo(const LoopHeader* header);
    void dump() const { root()->dump(); }
    const LoopLeaf* cfg_node2leaf(const CFGNode* cfg_node) const { return find(map_, cfg_node); }
    const LoopHeader* cfg_node2header(const CFGNode* cfg_node) const;

private:
    LoopLeaf* cfg_node2leaf(const CFGNode* cfg_node) { return map_[cfg_node]; }

    const CFGView<true>& cfg_;
    CFGNodeMap<LoopLeaf*> map_;
    Array<LoopLeaf*> dfs_leaves_;
    AutoPtr<LoopHeader> root_;

    friend class LoopTreeBuilder;
};

}

#endif
