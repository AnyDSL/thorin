#include "anydsl/analyses/domtree.h"

#include <limits>
#include <queue>

#include "anydsl/lambda.h"

#include "anydsl/analyses/scope.h"

namespace anydsl {

//------------------------------------------------------------------------------

DomNode::DomNode(const Lambda* lambda) 
    : lambda_(lambda) 
    , idom_(0)
{
    lambda->scratch.ptr = this;
}

DomNode::~DomNode() {
    for_all (child, children_)
        delete child;
}

//------------------------------------------------------------------------------

DomTree::DomTree(size_t size, const DomNode* root)
    : size_(size)
    , root_(root)
    , bfs_(size)
{
    size_t i = 0;
    // perform a breadth-first-traversal of the dom-tree
    std::queue<const DomNode*> q;
    q.push(root);

    while (!q.empty()) {
        const DomNode* node = q.front();
        q.pop();

        bfs_[i++] = node;

        for_all (child, node->children())
            q.push(child);
    }
}

bool DomTree::dominates(const DomNode* a, const DomNode* b) {
    while (a != b && !b->entry()) 
        b = b->idom();

    return a == b;
}

//------------------------------------------------------------------------------

class DomBuilder {
public:

    DomBuilder(const Lambda* entry, const LambdaSet& scope)
        : entry(entry)
        , scope(scope)
        , index2node(scope.size())
    {
        anydsl_assert(contains(entry), "entry not contained in scope");
    }


    size_t num() const { return index2node.size(); }
    bool contains(const Lambda* lambda) { return scope.find(lambda) != scope.end(); }
    static DomNode* node(const Lambda* lambda) { return (DomNode*) lambda->scratch.ptr; }

    DomTree build();
    DomNode* intersect(DomNode* i, DomNode* j);
    size_t number(const Lambda* cur, size_t i);

    const Lambda* entry;
    const LambdaSet& scope;
    Array<DomNode*> index2node;
};

DomTree DomBuilder::build() {
    // mark all nodes as unnumbered
    for_all (lambda, scope)
        lambda->scratch.ptr = 0;

    // mark all nodes in post-order
    size_t num2 = number(entry, 0);
    DomNode* entry_node = node(entry);
    anydsl_assert(num2 == num(), "bug in numbering -- maybe scope contains unreachable blocks?");
    anydsl_assert(num() - 1 == entry_node->index(), "bug in numbering");

    // map entry to entry, all other are set to 0 by the DomNode constructor
    entry_node->idom_ = entry_node;

    for (bool changed = true; changed;) {
        changed = false;

        // for all lambdas in reverse post-order except start node
        for (size_t i = num() - 2; i != size_t(-1); --i) {
            DomNode* cur = index2node[i];

            // for all predecessors of cur
            DomNode* new_idom = 0;
            for_all (pred, cur->lambda()->preds()) {
                if (contains(pred)) {
                    if (DomNode* other_idom = node(pred)->idom_) {
                        if (!new_idom)
                            new_idom = node(pred);// pick first processed predecessor of cur
                        else
                            new_idom = intersect(other_idom, new_idom);
                    }
                }
            }

            assert(new_idom);

            if (cur->idom() != new_idom) {
                cur->idom_ = new_idom;
                changed = true;
            }
        }
    }

    // add children -- thus iterate over all nodes except entry
    for (size_t i = 0, e = num() - 1; i != e; ++i) {
        const DomNode* node = index2node[i];
        node->idom_->children_.push_back(node);
    }

    return DomTree(num(), entry_node);
}

size_t DomBuilder::number(const Lambda* cur, size_t i) {
    DomNode* node = new DomNode(cur);

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (contains(succ) && !succ->scratch.ptr)
            i = number(succ, i);
    }

    node->index_ = i;
    index2node[i] = node;

    return i + 1;
}

DomNode* DomBuilder::intersect(DomNode* i, DomNode* j) {
    while (i->index() != j->index()) {
        while (i->index() < j->index()) 
            i = i->idom_;
        while (j->index() < i->index()) 
            j = j->idom_;
    }

    return i;
}

//------------------------------------------------------------------------------

DomTree calc_domtree(const Lambda* entry) {
    LambdaSet scope = find_scope(entry);
    return calc_domtree(entry, scope);
}

DomTree calc_domtree(const Lambda* entry, const LambdaSet& scope) {
    DomBuilder builder(entry, scope);
    return builder.build();
}

//------------------------------------------------------------------------------

} // namespace anydsl
