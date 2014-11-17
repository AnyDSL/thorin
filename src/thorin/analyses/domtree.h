#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

class CFNode;
template<bool> class CFG;

class DomNode {
public:
    explicit DomNode(const CFNode* cfg_node)
        : cfg_node_(cfg_node)
    {}

    const CFNode* cfg_node() const { return cfg_node_; }
    Lambda* lambda() const;
    const DomNode* idom() const { return idom_; }
    const std::vector<const DomNode*>& children() const { return children_; }
    size_t num_children() const { return children_.size(); }
    bool entry() const { return idom_ == this; }
    int depth() const { return depth_; }
    size_t max_rpo_id() const;
    void dump() const;

private:
    const CFNode* cfg_node_;
    DomNode* idom_ = nullptr;
    AutoVector<const DomNode*> children_;
    int depth_;
    size_t max_rpo_id_;

    template<bool> friend class DomTreeBase;
};

template<bool forward>
class DomTreeBase {
public:
    DomTreeBase(const DomTreeBase&) = delete;
    DomTreeBase& operator= (DomTreeBase) = delete;

    explicit DomTreeBase(const CFG<forward>&);

    const CFG<forward>& cfg() const { return cfg_; }
    size_t size() const { return nodes_.size(); }
    size_t rpo_id(const DomNode* n) const { return cfg().rpo_id(n->cfg_node()); }
    const DomNode* root() const { return root_; }
    /// Returns the least common ancestor of \p i and \p j.
    const DomNode* lca(const DomNode* i, const DomNode* j) const {
        return const_cast<DomTreeBase*>(this)->_lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j));
    }
    const DomNode* lookup(const CFNode* n) const { return nodes_[cfg().rpo_id(n)]; }
    void dump() const { root()->dump(); }

private:
    void create();
    size_t postprocess(DomNode* n, int depth);
    DomNode* _lca(DomNode*, DomNode*);
    DomNode*& _lookup(const CFNode* n) { return nodes_[cfg().rpo_id(n)]; }

    const CFG<forward>& cfg_;
    AutoPtr<DomNode> root_;
    std::vector<DomNode*> nodes_;
};

}

#endif
