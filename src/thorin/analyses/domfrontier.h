#ifndef THORIN_ANALYSES_DOMFRONTIER_H
#define THORIN_ANALYSES_DOMFRONTIER_H

#include "thorin/analyses/cfg.h"

namespace thorin {

/**
 * A Dominance Frontier Graph.
 * The template parameter @p forward determines whether to compute
 * regular dominance frontiers or post-dominance frontiers (i.e. control dependence).
 * This template parameter is associated with @p CFG's @c forward parameter.
 * See Cooper et al, 2001. A Simple, Fast Dominance Algorithm: http://www.cs.rice.edu/~keith/EMBED/dom.pdf
 */
template<bool forward>
class DomFrontierBase {
public:
    DomFrontierBase(const DomFrontierBase &) = delete;
    DomFrontierBase& operator=(DomFrontierBase) = delete;

    explicit DomFrontierBase(const CFG<forward> &cfg)
        : cfg_(cfg)
        , preds_(cfg)
        , succs_(cfg)
    {
        create();
    }

    const CFG<forward>& cfg() const { return cfg_; }
    const std::vector<const CFNode*>& preds(const CFNode* n) const { return preds_[n]; }
    const std::vector<const CFNode*>& succs(const CFNode* n) const { return succs_[n]; }

private:
    void create();
    void link(const CFNode* src, const CFNode* dst) {
        succs_[src].push_back(dst);
        preds_[dst].push_back(src);
    }

    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<std::vector<const CFNode*>> preds_;
    typename CFG<forward>::template Map<std::vector<const CFNode*>> succs_;
};

typedef DomFrontierBase<true>  DomFrontiers;
typedef DomFrontierBase<false> ControlDeps;

}

#endif
