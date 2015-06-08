#ifndef THORIN_ANALYSES_DFG_H
#define THORIN_ANALYSES_DFG_H

#include <vector>
#include "thorin/analyses/cfg.h"
#include "thorin/util/autoptr.h"

namespace thorin {

/**
 * @brief A Dominance Frontier Graph.
 *
 * The template parameter @p forward determines whether to compute regular dominance frontiers or post-dominance
 * frontiers (i.e. control dependence).
 * This template parameter is associated with @p CFG's @c forward parameter.
 */
template<bool forward>
class DFGBase {
public:
    class Node {
    private:
        explicit Node(const CFNode *cf_node)
            : cf_node_(cf_node)
        {}

    public:
        const CFNode* cf_node() const { return cf_node_; }
        const std::vector<const Node*>& preds() const { return preds_; }
        const std::vector<const Node*>& succs() const { return succs_; }
        void dump() const { /* TODO */ }

    private:
        const CFNode *cf_node_;
        mutable AutoVector<const Node*> preds_;
        mutable AutoVector<const Node*> succs_;

        friend class DFGBase<forward>;
    };

    DFGBase(const DFGBase &) = delete;
    DFGBase& operator=(DFGBase) = delete;

    explicit DFGBase(const CFG<forward> &cfg)
        : cfg_(cfg)
    {
        create();
    }

    const CFG<forward>& cfg() const { return cfg_; }
    size_t index(const Node* n) const { return cfg().index(n->cf_node()); }
    const Node* operator[](const CFNode* n) const { return nodes_[n]; }
    void dump() const { /* TODO */ }

private:
    void create();

    const CFG<forward> &cfg_;
    typename CFG<forward>::template Map<const Node*> nodes_;
};

typedef DFGBase<true>  DFG; /* Dominance Frontier Graph */
typedef DFGBase<false> CDG; /* Control Dependence Graph */
typedef DFG::Node DFNode;
typedef CDG::Node CDNode;

}

#endif
