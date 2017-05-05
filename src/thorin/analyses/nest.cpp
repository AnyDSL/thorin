#include "thorin/analyses/nest.h"

#include <queue>

#include "thorin/continuation.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class NestBuilder {
public:
    NestBuilder(const Scope& scope)
        : scope_(scope)
    {}

    const Scope& scope() { return scope_; }
    std::unique_ptr<const Nest::Node> run() {
        auto root = std::make_unique<const Nest::Node>(scope().entry(), nullptr, 0);
        for (auto continuation : scope().continuations())
            def2node(continuation);
        return root;
    }

private:
    const Nest::Node* def2node(const Def* def);

    const Scope& scope_;
    DefMap<const Nest::Node*> def2node_;
};

const Nest::Node* NestBuilder::def2node(const Def* def) {
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

//------------------------------------------------------------------------------

Nest::Nest(const Scope& scope)
    : scope_(scope)
    , root_(run())
    , top_down_(scope.size())
{}

std::unique_ptr<const Nest::Node> Nest::run() {
    auto root = NestBuilder(scope()).run();

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

}
