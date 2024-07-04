#ifndef THORIN_ANALYSES_LOOPTREE_H
#define THORIN_ANALYSES_LOOPTREE_H

#include <vector>
#include <sstream>

#include "thorin/analyses/cfg.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"

namespace thorin {

template<bool> class LoopTreeBuilder;

/**
 * Calculates a loop nesting forest rooted at @p root_.
 * The implementation uses Steensgard's algorithm.
 * Check out G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999, for more information.
 */
template<bool forward>
class LoopTree {
public:
    class Head;

    /**
    * Represents a node of a loop nesting forest.
    * Please refer to G. Ramalingam, "On Loops, Dominators, and Dominance Frontiers", 1999
    * for an introduction to loop nesting forests.
    * A @p Base consists of a set of header @p CFNode%s.
    * The header CFNode%s are the set of CFNode%s not dominated by any other @p CFNode within the loop.
    * The root node is a @p Head without any CFNode%s but further children and @p depth_ -1.
    * Thus, the forest is pooled into a tree.
    */
    class Base : public RuntimeCast<Base>, public Streamable<LoopTree<forward>::Base> {
    public:
        enum class Node { Head, Leaf };

        Base(Node node, Head* parent, int depth, const std::vector<const CFNode*>&);
        virtual ~Base() {}

        Node node() const { return node_; }
        int depth() const { return depth_; }
        const Head* parent() const { return parent_; }
        ArrayRef<const CFNode*> cf_nodes() const { return cf_nodes_; }
        size_t num_cf_nodes() const { return cf_nodes().size(); }
        virtual Stream& stream(Stream&) const = 0;

    protected:
        Node node_;
        Head* parent_;
        std::vector<const CFNode*> cf_nodes_;
        int depth_;
    };

    /// A Head owns further nodes as children.
    class Head : public Base {
    private:
        Head(Head* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
            : Base(Node, parent, depth, cf_nodes)
        {}

    public:
        ArrayRef<std::unique_ptr<Base>> children() const { return children_; }
        const Base* child(size_t i) const { return children_[i].get(); }
        size_t num_children() const { return children().size(); }
        bool is_root() const { return Base::parent_ == 0; }
        Stream& stream(Stream&) const override;

        static constexpr auto Node = Base::Node::Head;

    private:
        std::vector<std::unique_ptr<Base>> children_;

        friend class Base;
        friend class LoopTreeBuilder<forward>;
    };

    /// A Leaf only holds a single @p CFNode and does not have any children.
    class Leaf : public Base {
    private:
        Leaf(size_t index, Head* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
            : Base(Node, parent, depth, cf_nodes)
            , index_(index)
        {
            assert(Leaf::num_cf_nodes() == 1);
        }

    public:
        const CFNode* cf_node() const { return Leaf::cf_nodes().front(); }
        /// Index of a DFS of the @p LoopTree's @p Leaf%s.
        size_t index() const { return index_; }
        Stream& stream(Stream&) const override;

        static constexpr auto Node = Base::Node::Leaf;

    private:
        size_t index_;

        friend class LoopTreeBuilder<forward>;
    };

    LoopTree(const LoopTree&) = delete;
    LoopTree& operator=(LoopTree) = delete;

    explicit LoopTree(const CFG<forward>& cfg);
    const CFG<forward>& cfg() const { return cfg_; }
    const Head* root() const { return root_.get(); }
    const Leaf* operator[](const CFNode* n) const { return find(leaves_, n); }

private:
    static void get_nodes(std::vector<const Base *>& nodes, const Base* node) {
        nodes.push_back(node);
        if (auto head = node->template isa<Head>()) {
            for (const auto& child : head->children())
                get_nodes(nodes, child.get());
        }
    }

    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<Leaf*> leaves_;
    std::unique_ptr<Head> root_;

    friend class LoopTreeBuilder<forward>;
};

}

#endif
