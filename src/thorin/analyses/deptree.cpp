#include "thorin/analyses/deptree.h"

#include "thorin/world.h"

namespace thorin {

void DepTree::run() {
    for (const auto& [_, nom] : world().externals())
        roots_.emplace_back(std::make_unique<Node>(nom));
}

DepTree::Node* DepTree::run(Def* nom) {
    if (auto node = nom2node_.lookup(nom)) return *node;

    auto node = std::make_unique<Node>(nom);
    nom2node_[nom] = node.get();
    return node->set_parent(run(nom, nom));
}

DepTree::Node* DepTree::run(Def* cur_nom, const Def* def) {
    if (auto new_nom = def->isa_nominal()) return run(new_nom);

    if (auto param = def->isa<Param>()) {
        if (param->nominal() != cur_nom) {
            return run(param->nominal());
        } else {
            return nullptr;
        }
    }

    DepTree::Node* result;
    for (auto op : def->extended_ops()) {
        auto tmp = run(cur_nom, op);
        result = result->depth() < tmp->depth() ? tmp : result;
    }

    return result;
}

}
