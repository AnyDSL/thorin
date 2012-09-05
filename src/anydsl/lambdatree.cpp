#include "anydsl/lambdatree.h"

#include "anydsl/lambda.h"
#include "anydsl/analyses/find_root_lambdas.h"

#include "world.h"

namespace anydsl {


/*
 * helper
 */

static LambdaNode* get_node(const Lambda* lambda) { return (LambdaNode*) lambda->scratch.ptr; }

LambdaNode::LambdaNode(const Lambda* lambda)
    : lambda_(lambda)
    , parent_(this)
{ 
    lambda_->scratch.ptr = this; 
}

LambdaNode::~LambdaNode() {
    for_all (child, children())
        delete child;
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
}

void build_lambda_forest(const World& world) {
    LambdaSet lambdas = world.lambdas();
    find_root_lambdas(lambdas);
    dom(lambdas);
    build_lambda_forest(lambdas);
}

static const Lambda* find_user(const Def* def) {
    if (const Lambda* lambda = def->isa<Lambda>())
        return lambda;

    for_all (use, def->uses())
        return find_user(use.def());
}

static LambdaNode* race(LambdaNode* cur, LambdaNode* a, LambdaNode* b) {
    if (cur == a || cur == b)
        return cur;

    for_all (pred, cur->lambda()->callers()) {
        LambdaNode* winner = race(get_node(pred), a, b);
        get_node(pred)->parent_ = winner;
        return winner;
    }
}

static void go_up(const Lambda* cur, LambdaNode* to) {
    LambdaNode* cur_node = get_node(cur);

    if (cur_node->top())
        cur_node->parent_ = to;
    else {
        race(cur_node, cur_node->parent(), to);
    }

}

void build_lambda_forest(const LambdaSet& lambdas) {
    anydsl_assert(!lambdas.empty(), "must not be empty");

    for_all (lambda, lambdas)
        new LambdaNode(lambda);

    // init with simple dependencies
    for_all (lambda, lambdas) {
        for_all (param, lambda->params())
            go_up(find_user(param), get_node(lambda));
    }

    for_all (lambda, lambdas) {
        LambdaNode* node = get_node(lambda);
        std::cout << lambda->debug << ": " << node->parent()->lambda()->debug << std::endl;
    }
}

} // namespace anydsl
