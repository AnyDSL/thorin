#include "anydsl/analyses/placement.h"

#include <queue>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/analyses/domtree.h"

namespace anydsl {

typedef boost::unordered_set<const PrimOp*> Done;

void insert(Done& done, std::vector<const PrimOp*>& primops, const Def* def) {
    std::queue<const Def*> q;
    q.push(def);

    // perform a breadth-first-traversal of the uses
    while (!q.empty()) {
        const Def* def = q.front();
        q.pop();

        if (def->isa<Lambda>())
            continue;

        if (const PrimOp* primop = def->isa<PrimOp>()) {
            if (done.find(primop) != done.end())
                continue;
            else {
                done.insert(primop);
                for_all (op, primop->ops())
                    if (const Literal* lit = op->isa<Literal>())
                        primops.push_back(lit);

                primops.push_back(primop);
            }
        }

        for_all (use, def->uses())
            q.push(use.def());
    }
}

Places place(const DomTree& tree) {
    Places places(tree.size());
    Done done;

    for (size_t i = tree.size() - 1; i != size_t(-1); --i) {
        const DomNode* node = tree.bfs(i);
        for_all (param, node->lambda()->params())
            insert(done, places[node->index()], param);
    }

    // now check all primops which we might have missed because they not depend on params
    for_all (node, tree.bfs())
        for_all (arg, node->lambda()->args())
            insert(done, places[node->index()], arg);

    return places;
}

} // namespace anydsl
