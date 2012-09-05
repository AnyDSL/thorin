#include "anydsl/analyses/domtree.h"

#include <limits>

#include "anydsl/lambda.h"
#include "anydsl/world.h"

#include "anydsl/analyses/scope.h"

namespace anydsl {

static inline bool contains(const LambdaSet& set, const Lambda* lambda) {
    return set.find(lambda) != set.end();
}

static size_t number(Array<const Lambda*>& index2lambda, const LambdaSet& scope, const Lambda* lambda, size_t i);
static size_t intersect(const Array<const Lambda*>& idom, const Lambda* l, const Lambda* m);

const DomNode* calc_domtree(const Lambda* entry) {
    LambdaSet scope = find_scope(entry);
    return calc_domtree(entry, scope);
}

void calc_domtree(const World& world) {
    for_all (lambda, world.lambdas())
        if (lambda->isExtern())
            calc_domtree(lambda);
}

const DomNode* calc_domtree(const Lambda* entry, const LambdaSet& scope) {
    anydsl_assert(contains(scope, entry), "entry not contained in scope");
    size_t num = scope.size();

    // mark all nodes as not numbered
    for_all (lambda, scope)
        lambda->scratch.index = std::numeric_limits<size_t>::max();

    Array<const Lambda*> index2lambda(num);
    Array<const Lambda*> idom(num);

    // mark all nodes in post-order
    size_t num2 = number(index2lambda, scope, entry, 0);
    anydsl_assert(num2 == num, "bug in numbering -- maybe scope contains unreachable blocks?");
    anydsl_assert(num - 1 == entry->scratch.index, "bug in numbering");

    for_all (&lambda, idom)
        lambda = 0;
    // map entry to entry, all other are set to 0 by the array constructor
    idom[entry->scratch.index] = entry;

    if (num > 1) {
        for (bool changed = true; changed;) {
            changed = false;

            // for all lambdas in reverse post-order except start node
            for (size_t i = num - 1; i --> 0; /* the C++ goes-to operator :) */) {
                const Lambda* cur = index2lambda[i];

                // for all predecessors of cur
                const Lambda* new_idom = 0;
                for_all (caller, cur->callers()) {
                    if (contains(scope, caller)) {
                        if (const Lambda* other_idom = idom[caller->scratch.index]) {
                            if (!new_idom)
                                new_idom = caller;// pick first processed predecessor of cur
                            else
                                new_idom = index2lambda[intersect(idom, other_idom, new_idom)];
                        }
                    }
                }

                assert(new_idom);

                if (idom[cur->scratch.index] != new_idom) {
                    idom[cur->scratch.index] = new_idom;
                    changed = true;
                }
            }
        }
    }

    std::cout << "---" << std::endl;
    std::cout << num << '/' << idom.size() << '/' << index2lambda.size() << std::endl;
    for (size_t i = 0; i < num; ++i)
        std::cout << index2lambda[i]->debug << " -> " << idom[i]->debug << std::endl;

    std::cout << "---" << std::endl;

    return 0;
}

static size_t number(Array<const Lambda*>& index2lambda, const LambdaSet& scope, const Lambda* cur, size_t i) {
    // mark as visited
    --cur->scratch.index;

    // for each successor in scope
    for_all (succ, cur->succ()) {
        if (contains(scope, succ) && succ->scratch.index == std::numeric_limits<size_t>::max())
            i = number(index2lambda, scope, succ, i);
    }

    std::cout << cur->debug << ": " << i << std::endl;
    cur->scratch.index = i;
    index2lambda[i] = cur;

    return i + 1;
}

static size_t intersect(const Array<const Lambda*>& idom, const Lambda* l, const Lambda* m) {
    size_t i = l->scratch.index;
    size_t j = m->scratch.index;

    while (i != j) {
        while (i < j) 
            i = idom[i]->scratch.index;
        while (j < i) 
            j = idom[j]->scratch.index;
    }

    return i;
}

} // namespace anydsl
