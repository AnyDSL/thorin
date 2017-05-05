#include "thorin/analyses/nest.h"

#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class TreeBuilder {
public:
    TreeBuilder(Scope& scope)
        : scope_(scope)
    {}

    Scope& scope() { return scope_; }
    std::unique_ptr<const Nest::Node> run() {
        auto root = std::make_unique<const Nest::Node>(scope().entry(), nullptr, 0);
        for (auto continuation : scope().continuations())
            def2node(continuation);
        return root;
    }

private:
    const Nest::Node* def2node(const Def* def);

    Scope& scope_;
    DefMap<const Nest::Node*> def2node_;
};

const Nest::Node* TreeBuilder::def2node(const Def* def) {
    auto i = def2node_.find(def);
    if (i != def2node_.end())
        return i->second;

    //if (auto param = def->isa<Param>())

    auto n = def2node(def->ops().front());
    for (auto op : def->ops().skip_front()) {
        auto m = def2node(op);
        n = n->depth() > m->depth() ? n : n;
    }

    return n;
}

}
