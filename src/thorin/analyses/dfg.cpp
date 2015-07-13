#include "dfg.h"

#include <iostream>
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"

namespace thorin {

template<bool forward>
DFGBase<forward>::~DFGBase() {
    for (auto node : nodes_)
        delete node;
}

template<bool forward>
void DFGBase<forward>::Node::dump() const {
    if (const auto out_node = cf_node_->isa<OutNode>())
        std::cout << "(" << out_node->context()->def()->unique_name() << ") ";
    std::cout << cf_node_->def()->unique_name();
    std::cout << " -> [";
    for (auto succ : succs()) {
        if (*succs().begin() != succ)
            std::cout << ", ";
        std::cout << succ->cf_node()->def()->unique_name();
    }
    std::cout << "]" << std::endl;
}

//------------------------------------------------------------------------------

template<bool forward>
void DFGBase<forward>::create() {
    const auto& domtree = cfg().domtree();

    for (const auto node : cfg().rpo())
        nodes_[node] = new Node(node);

    // compute the dominance frontier of each node as described in Cooper et al.
    for (const auto node : cfg().body()) {
        const auto dfnode = (*this)[node];
        const auto& preds = cfg().preds(node);
        if (preds.size() > 1) {
            const auto idom = domtree[node]->idom()->cf_node();
            for (const auto pred : preds) {
                auto runner = pred;
                while (runner != idom) {
                    auto dfrunner = (*this)[runner];
                    dfnode->succs_.push_back(dfrunner);
                    dfrunner->preds_.push_back(dfnode);
                    runner = domtree[runner]->idom()->cf_node();
                }
            }
        }
    }
}

template<bool forward>
void DFGBase<forward>::dump() const {
    for (auto node : nodes_)
        node->dump();
}

template class DFGBase<true>;
template class DFGBase<false>;

}
