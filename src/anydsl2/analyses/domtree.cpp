#include "anydsl2/analyses/domtree.h"

#include <limits>
#include <queue>

#include "anydsl2/lambda.h"

#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

DomNode::DomNode(Lambda* lambda) 
    : lambda_(lambda) 
    , idom_(0)
{}

int DomNode::depth() const {
    int result = 0;

    for (const DomNode* i = this; !i->entry(); i = i->idom())
        ++result;

    return result;
};

size_t DomNode::sid() const { return lambda()->sid(); }

//------------------------------------------------------------------------------

DomTree::~DomTree() {
    for_all (node, nodes_)
        delete node;
}

void DomTree::create() {
    for_all (lambda, scope_.rpo())
        nodes_[lambda->sid()] = new DomNode(lambda);

    // map entry to entry, all other are set to 0 by the Array constructor
    DomNode* entry_node = lookup(scope_.entry());
    entry_node->idom_ = entry_node;

    for (bool changed = true; changed;) {
        changed = false;

        // for all lambdas in reverse post-order except entry node
        for_all (lambda, scope_.rpo().slice_back(1)) {
            DomNode* cur = lookup(lambda);

            // for all predecessors of cur
            DomNode* new_idom = 0;
            for_all (pred, scope().preds(lambda)) {
                if (DomNode* other_idom = lookup(pred)->idom_) {
                    if (!new_idom)
                        new_idom = lookup(pred);// pick first processed predecessor of cur
                    else
                        new_idom = lca(other_idom, new_idom);
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
    for_all (lambda, scope_.rpo().slice_back(1)) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

DomNode* DomTree::lca(DomNode* i, DomNode* j) {
    while (i->sid() != j->sid()) {
        while (i->sid() < j->sid()) j = j->idom_;
        while (j->sid() < i->sid()) i = i->idom_;
    }

    return i;
}

const DomNode* DomTree::lca(ArrayRef<const DomNode*> nodes) {
    assert(!nodes.empty());
    const DomNode* lca_node = nodes.front();
    for_all (n, nodes.slice_back(1))
        lca_node = lca(lca_node, n);

    return lca_node;
}

const DomNode* DomTree::node(Lambda* lambda) const { return nodes_[lambda->sid()]; }
DomNode* DomTree::lookup(Lambda* lambda) const { return nodes_[lambda->sid()]; }
size_t DomTree::size() const { return scope_.size(); }
const DomNode* DomTree::entry() const { return node(scope_.entry()); }

bool DomTree::dominates(const DomNode* a, const DomNode* b) {
    while (a != b && !b->entry()) 
        b = b->idom();

    return a == b;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
