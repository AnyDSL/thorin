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
}

template<bool forward>
void DFGBase<forward>::dump() const {
    for (auto &node : nodes_)
        node.dump();
}

}
