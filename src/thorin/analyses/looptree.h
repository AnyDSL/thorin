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
    struct Edge {
        Edge() {}
        Edge(const CFNode* src, const CFNode* dst, int levels)
            : src_(src)
            , dst_(dst)
            , levels_(levels)
        {}

        const CFNode* src() const { return src_; }
        const CFNode* dst() const { return dst_; }
        int levels() const { return levels_; }
        void dump();

    private:
        const CFNode* src_;
        const CFNode* dst_;
        int levels_;
    };

    explicit LoopHeader(const F_CFG& cfg, LoopHeader* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
        : LoopNode(parent, depth, cf_nodes)
        , dfs_begin_(0)
        , dfs_end_(-1)
        , headers_(cfg)
        , preheaders_(cfg)
        , latches_(cfg)
        , exits_(cfg)
        , exitings_(cfg)
    {}

    ArrayRef<LoopNode*> children() const { return children_; }
    const LoopNode* child(size_t i) const { return children_[i]; }
    size_t num_children() const { return children().size(); }
    const std::vector<Edge>& entry_edges() const { return entry_edges_; }
    const std::vector<Edge>& exit_edges() const { return exit_edges_; }
    const std::vector<Edge>& back_edges() const { return back_edges_; }
    /// Set of @p CFNode%s not dominated by any other @p CFNode within the loop.
    const F_CFG::Set& headers() const { return headers_; }
    /// Set of @p CFNode dominating the loop. They are not within the loop.
    const F_CFG::Set& preheaders() const { return preheaders_; }
    /// Set of @p CFNode%s which jump to one of the headers.
    const F_CFG::Set& latches() const { return latches_; }
    /// Set of @p CFNode%s which jump out of the loop.
    const F_CFG::Set& exitings() const { return exitings_; }
    /// Set of @p CFNode%s jumped to via exiting @p CFNode%s.
    const F_CFG::Set& exits() const { return exits_; }
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
    F_CFG::Set headers_;
    F_CFG::Set preheaders_;
    F_CFG::Set latches_;
    F_CFG::Set exits_;
    F_CFG::Set exitings_;

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
    size_t cf2dfs(const CFNode* cf_node) const { return cf2leaf(cf_node)->dfs_id(); }
    bool contains(const LoopHeader* header, const CFNode*) const;
    ArrayRef<const LoopLeaf*> loop(const LoopHeader* header) {
        return ArrayRef<const LoopLeaf*>(dfs_leaves_.data() + header->dfs_begin(), header->dfs_end() - header->dfs_begin());
    }
    Array<const CFNode*> loop_cf_nodes(const LoopHeader* header);
    Array<const CFNode*> loop_cf_nodes_in_rpo(const LoopHeader* header);
    void dump() const { root()->dump(); }
    const LoopLeaf* cf2leaf(const CFNode* n) const { return find(cf2leaf_, n); }
    const LoopHeader* cf2header(const CFNode*) const;

private:
    const F_CFG& cfg_;
    F_CFG::Map<LoopLeaf*> cf2leaf_;
    Array<LoopLeaf*> dfs_leaves_; // TODO use IndexMap
    AutoPtr<LoopHeader> root_;

    friend class LoopTreeBuilder;
};

}

#endif
