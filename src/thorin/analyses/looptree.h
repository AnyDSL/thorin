#ifndef ANALYSES_LOOPS_H
#define ANALYSES_LOOPS_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"

namespace thorin {

class Lambda;
class Scope;
class World;

struct Edge {
    Edge() {}
    Edge(Lambda* src, Lambda* dst, int levels) 
        : src_(src)
        , dst_(dst)
        , levels_(levels)
    {}

    Lambda* src() const { return src_; }
    Lambda* dst() const { return dst_; }
    int levels() const { return levels_; }
    void dump();

private:
    Lambda* src_;
    Lambda* dst_;
    int levels_;
};

class LoopHeader;

/**
 * Represents a node of a loop nesting forest.
 * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
 * for an introduction to loop nesting forests.
 * A \p LoopNode consists of a set of header \p Lambda%s.
 * The header lambdas are the set of lambdas not dominated by any other lambda within the loop.
 * The root node is a \p LoopHeader without any lambdas but further \p LoopNode children and \p depth_ -1.
 * Thus, the forest is pooled into a tree.
 */
class LoopNode : public MagicCast<LoopNode> {
public:
    LoopNode(LoopHeader* parent, int depth, const std::vector<Lambda*>& lambdas);

    int depth() const { return depth_; }
    const LoopHeader* parent() const { return parent_; }
    ArrayRef<Lambda*> lambdas() const { return lambdas_; }
    size_t num_lambdas() const { return lambdas().size(); }
    virtual void dump() const = 0;

protected:
    std::ostream& indent() const;

    LoopHeader* parent_;
    int depth_;
    std::vector<Lambda*> lambdas_;
};

/// A LoopHeader owns further \p LoopNode%s as children.
class LoopHeader : public LoopNode {
public:
    explicit LoopHeader(LoopHeader* parent, int depth, const std::vector<Lambda*>& lambdas)
        : LoopNode(parent, depth, lambdas)
        , dfs_begin_(0)
        , dfs_end_(-1)
    {}

    ArrayRef<LoopNode*> children() const { return children_; }
    const LoopNode* child(size_t i) const { return children_[i]; }
    size_t num_children() const { return children().size(); }
    const std::vector<Edge>& entry_edges() const { return entry_edges_; }
    const std::vector<Edge>& exit_edges() const { return exit_edges_; }
    const std::vector<Edge>& back_edges() const { return back_edges_; }
    /// Set of lambdas not dominated by any other lambda within the loop. Same as \p lambdas() as \p LambdaSet.
    const LambdaSet& headers() const { return headers_; }
    /// Set of lambdas dominating the loop. They are not within the loop.
    const LambdaSet& preheaders() const { return preheaders_; }
    /// Set of lambdas which jump to one of the headers.
    const LambdaSet& latches() const { return latches_; }
    /// Set of lambdas which jump out of the loop.
    const LambdaSet& exitings() const { return exitings_; }
    /// Set of lambdas jumped to via exiting lambdas.
    const LambdaSet& exits() const { return exits_; }
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
    LambdaSet headers_;
    LambdaSet preheaders_;
    LambdaSet latches_;
    LambdaSet exits_;
    LambdaSet exitings_;

    friend class LoopNode;
    friend class LoopTreeBuilder;
};

class LoopLeaf : public LoopNode {
public:
    explicit LoopLeaf(size_t dfs_index, LoopHeader* parent, int depth, const std::vector<Lambda*>& lambdas)
        : LoopNode(parent, depth, lambdas)
        , dfs_index_(dfs_index)
    {
        assert(num_lambdas() == 1);
    }

    Lambda* lambda() const { return lambdas().front(); }
    size_t dfs_index() const { return dfs_index_; }
    virtual void dump() const;

private:
    size_t dfs_index_;
};

/**
 * Calculates a loop nesting forest rooted at \p root_.
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
class LoopTree {
public:
    explicit LoopTree(const Scope& scope);

    const Scope& scope() const { return scope_; }
    const LoopHeader* root() const { return root_; }
    int depth(Lambda* lambda) const { return lambda2leaf(lambda)->depth(); }
    size_t lambda2dfs(Lambda* lambda) const { return lambda2leaf(lambda)->dfs_index(); }
    bool contains(const LoopHeader* header, Lambda* lambda) const;
    ArrayRef<const LoopLeaf*> loop(const LoopHeader* header) {
        return ArrayRef<const LoopLeaf*>(dfs_leaves_.data() + header->dfs_begin(), header->dfs_end() - header->dfs_begin());
    }
    Array<Lambda*> loop_lambdas(const LoopHeader* header);
    Array<Lambda*> loop_lambdas_in_rpo(const LoopHeader* header);
    void dump() const { root()->dump(); }
    const LoopLeaf* lambda2leaf(Lambda* lambda) const { return map_.find(lambda); }
    const LoopHeader* lambda2header(Lambda* lambda) const;

private:
    LoopLeaf* lambda2leaf(Lambda* lambda) { return map_[lambda]; }

    const Scope& scope_;
    LambdaMap<LoopLeaf*> map_;
    Array<LoopLeaf*> dfs_leaves_;
    AutoPtr<LoopHeader> root_;

    friend class LoopTreeBuilder;
};

} // namespace thorin

#endif // ANALYSES_LOOPS_H
