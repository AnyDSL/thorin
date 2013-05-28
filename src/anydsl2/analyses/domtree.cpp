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

//------------------------------------------------------------------------------

DomTree::DomTree(const Scope& scope, bool is_postdomtree)
    : scope_(scope)
    , nodes_(size())
    , is_postdomtree_(is_postdomtree)
{
    if (is_postdomtree)
        create<true>();
    else
        create<false>();
}

DomTree::~DomTree() {
    for_all (node, nodes_)
        delete node;
}

template<bool post>
void DomTree::create() {
    for_all (lambda, scope_.rpo())
        nodes_[index(lambda)] = new DomNode(lambda);

    // Map entries' initial idoms to themselves.
    for_all (entry,  post ? scope_.exits() : scope_.entries()) {
        DomNode* entry_node = lookup(entry);
        entry_node->idom_ = entry_node;
    }

    // All others' idoms are set to their first found dominating pred
    for_all (lambda, post ? scope_.backwards_body() : scope_.body()) {
        for_all (pred, post ? scope_.succs(lambda) : scope_.preds(lambda)) {
            if (index(pred) < index(lambda)) {
                lookup(lambda)->idom_ = lookup(pred);
                goto outer_loop;
            }
        }
        ANYDSL2_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for_all (lambda, post ? scope_.backwards_body() : scope_.body()) {
            DomNode* lambda_node = lookup(lambda);

            DomNode* new_idom = 0;
            for_all (pred, post ? scope_.succs(lambda) : scope_.preds(lambda)) {
                DomNode* pred_node = lookup(pred);
                assert(pred_node);
                new_idom = new_idom ? lca(new_idom, pred_node) : pred_node;
            }
            assert(new_idom);
            if (lambda_node->idom() != new_idom) {
                lambda_node->idom_ = new_idom;
                changed = true;
            }
        }
    }

    // add children
    for_all (lambda, post ? scope_.backwards_body() : scope_.body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

DomNode* DomTree::lca(DomNode* i, DomNode* j) {
    while (index(i) != index(j)) {
        while (index(i) < index(j)) j = j->idom_;
        while (index(j) < index(i)) i = i->idom_;
    }

    return i;
}

size_t DomTree::size() const { return scope_.size(); }

} // namespace anydsl2
