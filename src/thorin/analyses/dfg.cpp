#include "dfg.h"

#include <iostream>

namespace thorin {

template<bool forward>
void DFGBase<forward>::Node::dump() const {
    cf_node().dump();
    std::cout << " -> [";
    for (auto &pred : preds()) {
        if (preds().begin() != pred)
            std::cout << ", ";
        pred.cf_node().dump();
    }
    std::cout << "]" << std::endl;
}

//------------------------------------------------------------------------------

template<bool forward>
void DFGBase<forward>::create() {
    auto &domtree = cfg().domtree();

    for (auto n : cfg().rpo())
        nodes_.emplace(nodes_.end(), n);

    // compute the dominance frontier of each node as described in Cooper et al.
    for (auto n : cfg().body()) {
        const auto &preds = cfg().preds(n);
        if (preds.size() > 1) {
            auto idom = domtree[n]->idom()->cf_node();
            for (auto pred : preds) {
                auto runner = pred;
                while (runner != idom) {
                    auto domrunner = domtree[runner];
                    /* TODO: add DFG edge */
                    runner = domrunner->idom()->cf_node();
                }
            }
        }
    }
}

template<bool forward>
void DFGBase<forward>::dump() const {
    for (auto &node : nodes_)
        node.dump();
}

}
