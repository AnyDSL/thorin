#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/analyses/cfg.h"

namespace thorin {

/**
 * @brief A Dominance Tree.
 *
 * The template parameter @p forward determines
 * whether a regular dominance tree (@c true) or a post-dominance tree (@c false) should be constructed.
 * This template parameter is associated with @p CFG's @c forward parameter.
 */
template<bool forward>
class DomTreeBase : public YComp {
public:
    DomTreeBase(const DomTreeBase&) = delete;
    DomTreeBase& operator= (DomTreeBase) = delete;

    explicit DomTreeBase(const CFG<forward>& cfg)
        : YComp(cfg.scope(), forward ? "domtree" : "post_domtree")
        , cfg_(cfg)
        , idoms_(cfg)
        , children_(cfg)
    {
        create();
    }
    static const DomTreeBase& create(const Scope& scope) { return scope.cfg<forward>().domtree(); }

    const CFG<forward>& cfg() const { return cfg_; }
    size_t index(const CFNode* n) const { return cfg().index(n); }
    const CFNode* root() const { return *idoms_.begin(); }
    const CFNodes& children(const CFNode* n) const { return children_[n]; }
    const CFNode* idom(const CFNode* n) const { return idoms_[n]; }
    const CFNode* lca(const CFNode* i, const CFNode* j) const; ///< Returns the least common ancestor of @p i and @p j.
    virtual void stream_ycomp(std::ostream& out) const override;

private:
    void create();

    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<const CFNode*> idoms_;
    typename CFG<forward>::template Map<std::vector<const CFNode*>> children_;
};

typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;

}

#endif
