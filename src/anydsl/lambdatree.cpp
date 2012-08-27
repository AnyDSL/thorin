#include "anydsl/lambdatree.h"

#include "anydsl/lambda.h"

#include "world.h"

namespace anydsl {

LambdaNode::~LambdaNode() {
    for_all (child, children())
        delete child;
}

static void depends_simple(const Def* def, LambdaSet* dep) {
    if (const Param* param = def->isa<Param>())
        dep->insert(param->lambda());
    else if (!def->isa<Lambda>()) {
        for_all (op, def->ops())
            depends_simple(op, dep);
    }
}

static void group(const LambdaSet& lambdas) {
    std::queue<const Lambda*> queue;
    LambdaSet inqueue;

    typedef boost::unordered_map<const Lambda*, LambdaSet*> DepMap;
    DepMap depmap;

    for_all (lambda, lambdas) {
        LambdaSet* dep = new LambdaSet();

        for_all (op, lambda->ops())
            depends_simple(op, dep);

        depmap[lambda] = dep;
        queue.push(lambda);
        inqueue.insert(lambda);
    }

    while (!queue.empty()) {
        const Lambda* lambda = queue.front();
        queue.pop();
        inqueue.erase(lambda);
        LambdaSet* dep = depmap[lambda];
        size_t old = dep->size();

        for_all (succ, lambda->succ()) {
            LambdaSet* succ_dep = depmap[succ];

            for_all (d, *succ_dep) {
                if (d != succ)
                    dep->insert(d);
            }
        }

        if (dep->size() != old) {
            for_all (caller, lambda->callers()) {
                if (inqueue.find(caller) == inqueue.end()) {
                    inqueue.insert(caller);
                    queue.push(caller);
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "grouping:" << std::endl;
    std::cout << "---------" << std::endl;
    for_all (p, depmap) {
        std::cout << p.first->debug << std::endl;

        for_all (l, *p.second)
            std::cout << "\t" << l->debug << std::endl;
    }

    for_all (p, depmap)
        delete p.second;
}

static void dom(const LambdaSet& lambdas) {
    std::queue<const Lambda*> queue;
    LambdaSet inqueue;

    typedef std::set<const Lambda*> OrderedSet;
    typedef boost::unordered_map<const Lambda*, OrderedSet*> DomMap;
    DomMap dommap;

    for_all (lambda, lambdas) {
        OrderedSet* dom = new OrderedSet();

        if (lambda->callers().empty())
            dom->insert(lambda);
        else 
            std::copy(lambdas.begin(), lambdas.end(), std::inserter(*dom, dom->begin()));

        dommap[lambda] = dom;
        queue.push(lambda);
        inqueue.insert(lambda);
    }

    while (!queue.empty()) {
        const Lambda* lambda = queue.front();
        queue.pop();
        inqueue.erase(lambda);
        OrderedSet* dom = dommap[lambda];
        OrderedSet* nset = 0;
        bool first = true;

        for_all (caller, lambda->callers()) {
            OrderedSet* caller_dom = dommap[caller];

            if (first) {
                nset = new OrderedSet(*caller_dom);
                first = false;
            } else {
                OrderedSet* result = new OrderedSet();
                std::set_intersection(nset->begin(), nset->end(), caller_dom->begin(), caller_dom->end(), 
                                      std::inserter(*result, result->begin()));
                delete nset;
                nset = result;
            }
        }

        if (nset) {
            nset->insert(lambda);

            if (*nset != *dom) {
                *dom = *nset;
                for_all (succ, lambda->succ()) {
                    if (inqueue.find(succ) == inqueue.end()) {
                        inqueue.insert(succ);
                        queue.push(succ);
                    }
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "dominators:" << std::endl;
    std::cout << "-----------" << std::endl;
    for_all (p, dommap) {
        std::cout << p.first->debug << std::endl;

        for_all (l, *p.second)
            std::cout << "\t" << l->debug << std::endl;
    }

    //for_all (p, dommap)
        //delete p.second;
}

const LambdaNode* build_lambda_tree(const World& world) {
    LambdaSet lambdas;

    for_all (def, world.defs())
        if (const Lambda* lambda = def->isa<Lambda>())
            lambdas.insert(lambda);

    group(lambdas);
    dom(lambdas);
}

} // namespace anydsl
