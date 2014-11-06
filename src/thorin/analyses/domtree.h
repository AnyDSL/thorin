#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

class CFGNode;
template<bool> class CFGView;

class DomNode {
public:
    explicit DomNode(const CFGNode* cfg_node)
        : cfg_node_(cfg_node)
    {}

    const CFGNode* cfg_node() const { return cfg_node_; }
    Lambda* lambda() const;
    const DomNode* idom() const { return idom_; }
    const std::vector<const DomNode*>& children() const { return children_; }
    size_t num_children() const { return children_.size(); }
    bool entry() const { return idom_ == this; }
    int depth() const { return depth_; }
    size_t max_rpo_id() const;
    void dump() const;

private:
    const CFGNode* cfg_node_;
    DomNode* idom_ = nullptr;
    AutoVector<const DomNode*> children_;
    int depth_;
    size_t max_rpo_id_;

    template<bool> friend class DomTreeBase;
};

template<bool forward>
class DomTreeBase {
public:
    explicit DomTreeBase(const CFGView<forward>&);

    const CFGView<forward>& cfg_view() const { return cfg_view_; }
    size_t size() const { return nodes_.size(); }
    size_t rpo_id(const DomNode* n) const { return cfg_view().rpo_id(n->cfg_node()); }
    const DomNode* root() const { return root_; }
    //int depth(const CFGNode* n) const { return lookup(n)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    //const CFGNode* lca(const CFGNode* i, const CFGNode* j) const { return lca(lookup(i), lookup(j))->cfg_node(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const {
        return const_cast<DomTreeBase*>(this)->_lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j));
    }
    //const CFGNode* idom(const CFGNode* n) const { return lookup(n)->idom()->cfg_node(); }
    const DomNode* lookup(const CFGNode* n) const { return nodes_[cfg_view().rpo_id(n)]; }
    //const DomNode* lookup(size_t rpo_id) const { return nodes_[rpo_id]; }
    void dump() const { root()->dump(); }

private:
    void create();
    size_t postprocess(DomNode* n, int depth);
    DomNode* _lca(DomNode*, DomNode*);
    DomNode*& _lookup(const CFGNode* n) { return nodes_[cfg_view().rpo_id(n)]; }

    const CFGView<forward>& cfg_view_;
    AutoPtr<DomNode> root_;
    std::vector<DomNode*> nodes_;
};

}

#endif
