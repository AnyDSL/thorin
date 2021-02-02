#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/analyses/cfg.h"

namespace thorin {

/**
 * A Dominance Tree.
 * The template parameter @p forward determines
 * whether a regular dominance tree (@c true) or a post-dominance tree (@c false) should be constructed.
 * This template parameter is associated with @p CFG's @c forward parameter.
 */
template<bool forward>
class DomTreeBase {
public:
    DomTreeBase(const DomTreeBase&) = delete;
    DomTreeBase& operator=(DomTreeBase) = delete;

    explicit DomTreeBase(const CFG<forward>& cfg)
        : cfg_(cfg)
        , children_(cfg)
        , idoms_(cfg)
        , depth_(cfg)
    {
        create();
        depth(root(), 0);
    }

    const CFG<forward>& cfg() const { return cfg_; }
    size_t index(const CFNode* n) const { return cfg().index(n); }
    const std::vector<const CFNode*>& children(const CFNode* n) const { return children_[n]; }
    const CFNode* root() const { return *idoms_.begin(); }
    const CFNode* idom(const CFNode* n) const { return idoms_[n]; }
    int depth(const CFNode* n) const { return depth_[n]; }
    const CFNode* least_common_ancestor(const CFNode* i, const CFNode* j) const;

private:
    void create();
    void depth(const CFNode* n, int i);

    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<std::vector<const CFNode*>> children_;
    typename CFG<forward>::template Map<const CFNode*> idoms_;
    typename CFG<forward>::template Map<int> depth_;
};

typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;

}

#endif
