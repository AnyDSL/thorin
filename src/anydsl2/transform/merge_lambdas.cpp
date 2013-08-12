#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/verify.h"

namespace anydsl2 {

class Merger {
public:
    Merger(World& world)
        : scope(world)
        , domtree(scope.domtree())
    {
        for (auto entry : scope.entries())
            merge(scope.domtree().node(entry));
    }

    void merge(const DomNode* n);
    const DomNode* dom_succ(const DomNode* n);
    World& world() { return scope.world(); }

    Scope scope;
    const DomTree& domtree;
};

const DomNode* Merger::dom_succ(const DomNode* n) { 
    ArrayRef<Lambda*> succs = scope.succs(n->lambda());
    const std::vector<const DomNode*>& children = n->children();
    return succs.size() == 1 && children.size() == 1 
        && succs.front() == children.front()->lambda() 
        && succs.front()->num_uses() == 1
        && succs.front() == n->lambda()->to()
        ? children.front() : 0;
}

void Merger::merge(const DomNode* n) {
    const DomNode* cur = n;
    for (const DomNode* next = dom_succ(cur); next != 0; cur = next, next = dom_succ(next)) {
        assert(cur->lambda()->num_args() == next->lambda()->num_params());
        for (size_t i = 0, e = cur->lambda()->num_args(); i != e; ++i)
            next->lambda()->param(i)->replace(cur->lambda()->arg(i));
        cur->lambda()->destroy_body();
    }

    if (cur != n)
        n->lambda()->jump(cur->lambda()->to(), cur->lambda()->args());

    for (auto child : cur->children())
        merge(child);
}

void merge_lambdas(World& world) { Merger merger(world); debug_verify(world); }

} // namespace anydsl2
