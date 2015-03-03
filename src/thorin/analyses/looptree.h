#ifndef THORIN_ANALYSES_LOOPTREE_H
#define THORIN_ANALYSES_LOOPTREE_H

#include <vector>

#include "thorin/analyses/cfg.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"

namespace thorin {

template<bool> class LoopTreeBuilder;

/**
 * @brief Calculates a loop nesting forest rooted at @p root_.
 * 
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
template<bool forward>
class LoopTree {
public:
    class Head;

    /**
    * @brief Represents a node of a loop nesting forest.
    *
    * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
    * for an introduction to loop nesting forests.
    * A @p Node consists of a set of header @p CFNode%s.
    * The header CFNode%s are the set of CFNode%s not dominated by any other @p CFNode within the loop.
    * The root node is a @p Head without any CFNode%s but further @p Node children and @p depth_ -1.
    * Thus, the forest is pooled into a tree.
    */
    class Node : public MagicCast<Node> {
    public:
        Node(Head* parent, int depth, const std::vector<const CFNode*>&);

        int depth() const { return depth_; }
        const Head* parent() const { return parent_; }
        ArrayRef<const CFNode*> cf_nodes() const { return cf_nodes_; }
        size_t num_cf_nodes() const { return cf_nodes().size(); }
        virtual void dump() const = 0;

    protected:
        std::ostream& indent() const;

        Head* parent_;
        int depth_;
        std::vector<const CFNode*> cf_nodes_;
    };

    /// A Head owns further @p Node%s as children.
    class Head : public Node {
    public:
        typedef Node Super;

        Head(Head* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
            : Super(parent, depth, cf_nodes)
        {}

        ArrayRef<Super*> children() const { return children_; }
        const Super* child(size_t i) const { return children_[i]; }
        size_t num_children() const { return children().size(); }
        bool is_root() const { return Super::parent_ == 0; }
        virtual void dump() const;

    private:
        AutoVector<Super*> children_;

        friend class Node;
        friend class LoopTreeBuilder<forward>;
    };

    class Leaf : public Node {
    public:
        typedef Node Super;

        explicit Leaf(size_t dfs_id, Head* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
            : Super(parent, depth, cf_nodes)
            , dfs_id_(dfs_id)
        {
            assert(Super::num_cf_nodes() == 1);
        }

        const CFNode* cf_node() const { return Super::cf_nodes().front(); }
        size_t dfs_id() const { return dfs_id_; }
        virtual void dump() const;

    private:
        size_t dfs_id_;
    };

    LoopTree(const LoopTree&) = delete;
    LoopTree& operator= (LoopTree) = delete;

    explicit LoopTree(const CFG<forward>& cfg);

    const CFG<forward>& cfg() const { return cfg_; }
    const Head* root() const { return root_; }
    int depth(const CFNode* n) const { return cf2leaf(n)->depth(); }
    size_t cf2dfs(const CFNode* n) const { return cf2leaf(n)->dfs_id(); }
    void dump() const { root()->dump(); }
    const Leaf* cf2leaf(const CFNode* n) const { return find(cf2leaf_, n); }
    const Head* cf2header(const CFNode*) const;

private:
    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<Leaf*> cf2leaf_;
    AutoPtr<Head> root_;

    friend class LoopTreeBuilder<forward>;
};

}

#endif
