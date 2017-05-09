#include "thorin/analyses/nest.h"

#include <queue>

#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Nest::Nest(const Scope& scope)
    : scope_(scope)
    , def2node_(scope.defs().capacity())
    , top_down_(scope.continuations().size())
    , root_(run())
{}

std::unique_ptr<const Nest::Node> Nest::run() {
    auto root = std::make_unique<const Nest::Node>(scope().entry(), nullptr, 0);
    def2node_[scope().entry()] = root.get();
    def2node_[scope().exit()] = root->bear(scope().exit());

    for (auto def : scope().defs())
        def2node(def);

    // now build top-down order
    std::queue<const Node*> queue;
    size_t i = 0;

    auto enqueue = [&](const Node* n) {
        queue.push(n);
        top_down_[i++] = n;
    };

    enqueue(root.get());

    while (!queue.empty()) {
        for (const auto& child : pop(queue)->children())
            enqueue(child.get());
    }

    assert(i == top_down().size());
    return root;
}

const Node* Nest::def2node(const Node* n, const Def* def) {
    auto i = def2node_.find(def);
    if (i != def2node_.end())
        return i->second;

    auto set = [&](const Node* n) {
        if (auto continuation = def->isa_continuation()) {
            def2node_[continuation] = n;
            for (auto param : continuation->params())
                def2node_[param] = n;
        }
    };

    // avoid cycles
    set(nullptr);

    const Node* n = nullptr;;
    if (auto param = def->isa<Param>()) {
        n = def2node(param->continuation());
        assert(n);
    } else {
        for (auto op : def->ops()) {
            if (scope().contains(op)) {
                if (auto m = def2node(op))
                    n = n ? (n->depth() > m->depth() ? n : m) : m;
            }
        }

        assert(n != nullptr);

        if (auto continuation = def->isa_continuation())
            n = n->bear(continuation);

        set(n);
    }

    return def2node_[def] = n;
}

}
