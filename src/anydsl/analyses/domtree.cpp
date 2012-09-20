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

    DomBuilder(const Scope& scope)
        : scope(scope)
        , index2node(scope.size())
    {}

    size_t size() const { return scope.size(); }
    static DomNode* node(const Lambda* lambda) { return (DomNode*) lambda->scratch.ptr; }

    DomTree build();
    DomNode* intersect(DomNode* i, DomNode* j);
    size_t number(const Lambda* cur, size_t i);

    const Scope& scope;
    Array<DomNode*> index2node;
};

DomTree DomBuilder::build() {
    for_all (lambda, scope.rpo()) {
        index2node[lambda->sid] = new DomNode(lambda);
        index2node[lambda->sid]->index_ = lambda->sid;
    }

    // map entry to entry, all other are set to 0 by the DomNode constructor
    DomNode* entry_node = (DomNode*) scope.entry()->scratch.ptr;
    entry_node->idom_ = entry_node;

    for (bool changed = true; changed;) {
        changed = false;

        // for all lambdas in reverse post-order except start node
        for_all (lambda, scope.rpo().slice_back(1)) {
            DomNode* cur = index2node[lambda->sid];

            // for all predecessors of cur
            DomNode* new_idom = 0;
            for_all (pred, scope.preds(cur->lambda())) {
                if (DomNode* other_idom = node(pred)->idom_) {
                    if (!new_idom)
                        new_idom = node(pred);// pick first processed predecessor of cur
                    else
                        new_idom = intersect(other_idom, new_idom);
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
    for_all (lambda, scope.rpo().slice_back(1)) {
        const DomNode* node = index2node[lambda->sid];
        node->idom_->children_.push_back(node);
    }

    return DomTree(size(), entry_node);
}

DomNode* DomBuilder::intersect(DomNode* i, DomNode* j) {
    while (i->index() != j->index()) {
        while (i->index() < j->index()) 
            j = j->idom_;
        while (j->index() < i->index()) 
            i = i->idom_;
    }

    return i;
}

//------------------------------------------------------------------------------

DomTree calc_domtree(const Scope& scope) {
    DomBuilder builder(scope);
    return builder.build();
}

int DomNode::depth() const {
    int result = 0;

    for (const DomNode* i = this; !i->entry(); i = i->idom())
        ++result;

    return result;
};

//------------------------------------------------------------------------------

} // namespace anydsl
