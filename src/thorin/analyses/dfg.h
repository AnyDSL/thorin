#ifndef THORIN_ANALYSES_DFG_H
#define THORIN_ANALYSES_DFG_H

#include <iostream>
#include <sstream>
#include <vector>

#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/graphs.h"

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
        explicit Node(const CFNode* cf_node)
            : cf_node_(cf_node)
        {}

    public:
        const CFNode* cf_node() const { return cf_node_; }
        const std::vector<const Node*>& preds() const { return preds_; }
        const std::vector<const Node*>& succs() const { return succs_; }
        void dump(std::ostream& os) const;
        void dump() const { dump(std::cerr); }

    private:
        const CFNode* cf_node_;
        mutable std::vector<const Node*> preds_;
        mutable std::vector<const Node*> succs_;

        friend class DFGBase<forward>;
    };

    DFGBase(const DFGBase &) = delete;
    DFGBase& operator=(DFGBase) = delete;

    explicit DFGBase(const CFG<forward> &cfg)
        : cfg_(cfg)
        , nodes_(cfg)
    {
        create();
    }

    ~DFGBase();

    const CFG<forward>& cfg() const { return cfg_; }
    size_t index(const Node* n) const { return cfg().index(n->cf_node()); }
    const Node* operator[](const CFNode* n) const { return nodes_[n]; }
    void dump(std::ostream& os) const;
    void dump() const { dump(std::cerr); }

        static void emit_scope(const Scope& scope, std::ostream& ostream = std::cout) {
            auto& dfg = scope.cfg<forward>().dfg();
            //DFGBase<forward>& dfg = nullptr;

            emit_ycomp(ostream, scope, range(dfg.nodes_.begin(), dfg.nodes_.end()),
                       [] (const Node* node) {
                           return range(node->succs().begin(), node->succs().end());
                       },
                       [] (const Node* node) {
                           std::stringstream stream;
                           if (auto out_node = node->cf_node_->template isa<OutNode>())
                               stream << "(" << out_node->context()->def()->unique_name() << ") ";
                           stream << node->cf_node_->def()->unique_name();

                           return std::make_pair(stream.str(), stream.str());
                       },
                       YComp_Orientation::TOP_TO_BOTTOM
            );
        }

        static void emit_world(const World& world, std::ostream& ostream = std::cout) {
            emit_ycomp(ostream, world, emit_scope);
        }

private:
    void create();

    const CFG<forward>& cfg_;
    typename CFG<forward>::template Map<const Node*> nodes_;
};

//------------------------------------------------------------------------------

typedef DFGBase<true>  DFG; /* Dominance Frontier Graph */
typedef DFGBase<false> CDG; /* Control Dependence Graph */
typedef DFG::Node DFNode;
typedef CDG::Node CDNode;

}

#endif
