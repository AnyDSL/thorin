#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/analyses/cfg.h"
#include "thorin/be/graphs.h"
#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/iterator.h"

namespace thorin {

class CFNode;
template<bool> class CFG;

/**
 * @brief A Dominance Tree.
 *
 * The template parameter @p forward determines 
 * whether a regular dominance tree (@c true) or a post-dominance tree (@c false) should be constructed.
 * This template parameter is associated with @p CFG's @c forward parameter.
 */
template<bool forward>
class DomTreeBase {
public:
    class Node {
    private:
        explicit Node(const CFNode* cf_node)
            : cf_node_(cf_node)
        {}

    public:
        const CFNode* cf_node() const { return cf_node_; }
        const Node* idom() const { return idom_; }
        const InNode* in_idom() const { return idom_->cf_node()->in_node(); }
        const std::vector<const Node*>& children() const { return children_; }
        size_t num_children() const { return children_.size(); }
        bool entry() const { return idom_ == this; }
        void dump() const { dump(0); }

    private:
        void dump(const int depth) const;

        const CFNode* cf_node_;
        mutable const Node* idom_ = nullptr;
        mutable AutoVector<const Node*> children_;

        template<bool> friend class DomTreeBase;
    };

    DomTreeBase(const DomTreeBase&) = delete;
    DomTreeBase& operator= (DomTreeBase) = delete;

    explicit DomTreeBase(const CFG<forward>& cfg)
        : cfg_(cfg)
        , nodes_(cfg)
    {
        create();
    }

        static void emit_scope(const Scope& scope, std::ostream& ostream = std::cout) {
            auto& domtree = scope.cfg<forward>().domtree();
            emit_ycomp(ostream, scope, range(domtree.nodes().begin(), domtree.nodes().end()),
                       [] (const Node* node) {
                           return range(node->children().begin(), node->children().end());
                       },
                       [] (const Node* node) {
                           auto id = std::to_string(node->cf_node()->def()->gid()) + "_" + std::to_string(node->cf_node()->in_node()->def()->gid());
                           std::string label;
                           if(auto out = node->cf_node()->template isa<thorin::OutNode>()) {
                               label += "(" + out->context()->def()->unique_name() + ") ";
                           }

                           label += node->cf_node()->def()->unique_name();
                           return std::make_pair(id, label);
                       },
                       YComp_Orientation::TOP_TO_BOTTOM
            );
        }

        static void emit_world(const World& world, std::ostream& ostream = std::cout) {
            emit_ycomp(ostream, world, emit_scope);
        }

    const CFG<forward>& cfg() const { return cfg_; }
    size_t index(const Node* n) const { return cfg().index(n->cf_node()); }
    ArrayRef<const Node*> nodes() const { return nodes_.array(); }
    const Node* root() const { return root_; }
    /// Returns the least common ancestor of @p i and @p j.
    const Node* lca(const Node* i, const Node* j) const;
    const Node* operator [] (const CFNode* n) const { return nodes_[n]; }
    void dump() const { root()->dump(); }

private:
    void create();

    const CFG<forward>& cfg_;
    AutoPtr<const Node> root_;
    typename CFG<forward>::template Map<const Node*> nodes_;
};

typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;
typedef DomTree::Node      DomNode;
typedef PostDomTree::Node  PostDomNode;

}

#endif