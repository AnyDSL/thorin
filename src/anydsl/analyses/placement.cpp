#include "anydsl/analyses/placement.h"

#include <queue>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/analyses/scope.h"

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

Places place(const Scope& scope) {
    Places places(scope.size());
    Done done;

    for (size_t i = scope.size() - 1; i != size_t(-1); --i) {
        Lambda* lambda = scope.rpo(i);
        for_all (param, lambda->params())
            insert(done, places[i], param);
    }

    return places;
}

} // namespace anydsl
