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

    // map entry to entry, all other are set to the first dominating pred
    DomNode* entry_node = lookup(scope_.entry());
    entry_node->idom_ = entry_node;

    for_all (lambda, scope_.rpo().slice_back(1)) {
        for_all (pred, scope().preds(lambda)) {
            if (pred->sid() < lambda->sid()) {
                lookup(lambda)->idom_ = lookup(pred);
                goto outer_loop;
            }
        }
        ANYDSL2_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        // for all lambdas in reverse post-order except entry node
        for_all (lambda, scope_.rpo().slice_back(1)) {
            DomNode* lambda_node = lookup(lambda);

            // for all predecessors of lambda
            DomNode* new_idom = 0;
            for_all (pred, scope().preds(lambda)) {
                DomNode* pred_node = lookup(pred);
                assert(pred_node);
                new_idom = new_idom ?  lca(new_idom, pred_node) : pred_node;
            }
            assert(new_idom);
            if (lambda_node->idom() != new_idom) {
                lambda_node->idom_ = new_idom;
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

const DomNode* DomTree::node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[lambda->sid()]; }
DomNode* DomTree::lookup(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[lambda->sid()]; }
size_t DomTree::size() const { return scope_.size(); }
const DomNode* DomTree::entry() const { return node(scope_.entry()); }

bool DomTree::dominates(const DomNode* a, const DomNode* b) const {
    while (a != b && !b->entry()) 
        b = b->idom();

    return a == b;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
