#include "thorin/analyses/nest.h"

#include <queue>

#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Nest::Nest(const Scope& scope)
    : scope_(scope)
    , def2node_(scope.defs().capacity())
    , top_down_(scope.continuations().size())
    , map_(round_to_power_of_2(scope.continuations().size()))
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
        map_[n->continuation()] = n;
    };

    enqueue(root.get());

    while (!queue.empty()) {
        for (const auto& child : pop(queue)->children())
            enqueue(child.get());
    }

    top_down_.shrink(i);
    return root;
}

const Nest::Node* Nest::def2node(const Def* def) {
    auto i = def2node_.find(def);
    if (i != def2node_.end())
        return i->second;

    auto set_continuation = [&](const Node* n) {
        auto continuation = def->as_continuation();
        def2node_[continuation] = n;
        for (auto param : continuation->params())
            def2node_[param] = n;
    };

    // avoid cycles
    if (def->isa<Continuation>())
        set_continuation(nullptr);

    const Node* n = nullptr;;
    if (auto param = def->isa<Param>())
        n = def2node(param->continuation());
    else {
        for (auto use : def->uses()) {
            if (scope().contains(use)) {
                if (auto m = def2node(use))
                    n = n ? (n->depth() > m->depth() ? n : m) : m;
            }
        }

        if (auto continuation = def->isa_continuation()) {
            n = n != nullptr ? n->bear(continuation) : nullptr;
            set_continuation(n);
        }
    }

    return def2node_[def] = n;
}

}
