#include "thorin/analyses/deptree.h"

#include "thorin/world.h"

namespace thorin {

static void merge(ParamSet& params, ParamSet&& other) {
    params.insert(other.begin(), other.end());
}

void DepTree::run() {
    for (const auto& [_, nom] : world().externals()) run(nom);
    adjust_depth(root_.get(), 0);
}

ParamSet DepTree::run(Def* nom) {
    if (auto node = nom2node_.lookup(nom)) {
        if (auto params = def2params_.lookup(nom))
            return *params;
        else
            return ParamSet();
    }

    auto node = std::make_unique<Node>(nom, stack_.size()+1);
    stack_.push_back(node.get());
    nom2node_[nom] = node.get();

    auto result = run(nom, nom);

    auto parent = root_.get();
    for (auto param : result) {
        auto n = nom2node_[param->nominal()];
        parent = n->depth() > parent->depth() ? n : parent;
    }
    node->set_parent(parent);

    stack_.pop_back();
    return result;
}

ParamSet DepTree::run(Def* cur_nom, const Def* def) {
    if (def->is_const())                                         return {};
    if (auto params = def2params_.lookup(def))                   return *params;
    if (auto nom    = def->isa_nominal(); nom && cur_nom != nom) return run(nom);

    ParamSet result;
    if (auto param = def->isa<Param>()) {
        result.emplace(param);
    } else {
        for (auto op : def->extended_ops())
            merge(result, run(cur_nom, op));

        if (cur_nom == def) result.erase(cur_nom->param());
    }

    return def2params_[def] = ParamSet(result);
}

void DepTree::adjust_depth(Node* node, size_t depth) {
    node->depth_ = depth;

    for (const auto& child : node->children())
        adjust_depth(child.get(), depth + 1);
}

}
