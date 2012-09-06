#include "anydsl/analyses/placement.h"

#include <queue>

#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/analyses/domtree.h"

namespace anydsl {

void insert(std::vector<const PrimOp*>& primops, const Def* def) {
    std::queue<const Def*> q;
    q.push(def);

    // perform a breadth-first-traversal of the uses
    while (!q.empty()) {
        const Def* def = q.front();
        q.pop();

        if (def->isa<Lambda>())
            continue;

        if (const PrimOp* primop = def->isa<PrimOp>())
            primops.push_back(primop);

        for_all (use, def->uses())
            q.push(use.def());
    }
}

Places place(const DomTree& tree) {
    Places places(tree.size());

    for_all (node, tree.bfs()) {
        for_all (param, node->lambda()->params())
            insert(places[node->index()], param);
    }

    return places;
}

} // namespace anydsl
