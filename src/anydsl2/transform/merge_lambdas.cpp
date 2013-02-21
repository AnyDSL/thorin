#include "anydsl2/transform/merge_lambdas.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/domtree.h"

namespace anydsl2 {

class Merger {
public:

    Merger(Lambda* entry)
        : scope(entry)
        , domtree(scope.domtree())
    {
        merge(domtree.entry());
    }

    void merge(const DomNode* n);
    const DomNode* dom_succ(const DomNode* n);
    World& world() { return scope.world(); }

    Scope scope;
    const DomTree& domtree;
};

const DomNode* Merger::dom_succ(const DomNode* n) { 
    ArrayRef<Lambda*> succs = scope.succs(n->lambda());
    const DomNodes& children = n->children();
    return succs.size() == 1 && children.size() == 1 && succs.front() == children.front()->lambda() && n->lambda()->to() == succs.front() ? children.front() : 0;
}

void Merger::merge(const DomNode* n) {
    const DomNode* i = n;
    for (const DomNode* next = dom_succ(i); next != 0; i = next, next = dom_succ(next)) {
        assert(i->lambda()->num_args() == next->lambda()->num_params());
        for_all2 (arg, i->lambda()->args(), param, next->lambda()->params())
            param->replace_all_uses_with(arg);
    }

    if (i != n)
        n->lambda()->jump(i->lambda()->to(), i->lambda()->args());

    for_all (child, i->children())
        merge(child);
}

void merge_lambdas(World& world) {
    for_all (top, find_root_lambdas(world)) {
        Merger merger(top);
    }
}

} // namespace anydsl2
