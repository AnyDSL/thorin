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

DomTree::DomTree(const Scope& scope, bool post)
    : scope_(scope)
    , nodes_(size())
{
    if (post)
        create_postdom();
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
        nodes_[lambda->sid()] = new DomNode(lambda);

    // map entry's initial idoms their entry,
    // all others' idoms are set to their first found dominating pred
    for_all (entry,  scope_.entries()) {
        DomNode* entry_node = lookup(entry);
        entry_node->idom_ = entry_node;
    }

    for_all (lambda, scope_.body()) {
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

        for_all (lambda, scope_.body()) {
            DomNode* lambda_node = lookup(lambda);

            // for all predecessors of lambda
            DomNode* new_idom = 0;
            for_all (pred, scope().preds(lambda)) {
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
    for_all (lambda, scope_.body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

void DomTree::create_postdom() {
#if 0
    for_all (lambda, scope_.rpo())
        nodes_[lambda->sid()] = new DomNode(lambda);

    // map entry's initial idoms their entry,
    // all others' idoms are set to their first found dominating pred
    for_all (entry,  scope_.exits()) {
        DomNode* entry_node = lookup(entry);
        entry_node->idom_ = entry_node;
    }

    for (ArrayRef<Lambda*>::const_reverse_iterator i_lambda = scope_.reverse_body().rbegin(), 
            e_lambda = scope_.reverse_body().rend(); i_lambda != e_lambda; ++i_lambda) {
        Lambda* lambda = *i_lambda;
        for_all (succ, scope().succs(lambda)) {
            if (succ->sid() > lambda->sid()) {
                lookup(lambda)->idom_ = lookup(succ);
                goto outer_loop;
            }
        }
        ANYDSL2_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for (ArrayRef<Lambda*>::const_reverse_iterator i_lambda = scope_.reverse_body().rbegin(), 
                e_lambda = scope_.reverse_body().rend(); i_lambda != e_lambda; ++i_lambda) {
            Lambda* lambda = *i_lambda;
            DomNode* lambda_node = lookup(lambda);

            // for all predecessors of lambda
            DomNode* new_idom = 0;
            for_all (succ, scope().succs(lambda)) {
                DomNode* succ_node = lookup(succ);
                assert(succ_node);
                new_idom = new_idom ? post_lca(new_idom, succ_node) : succ_node;
            }
            assert(new_idom);
            if (lambda_node->idom() != new_idom) {
                lambda_node->idom_ = new_idom;
                changed = true;
            }
        }
    }

    // add children
    for_all (lambda, scope_.body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
#endif
}

DomNode* DomTree::lca(DomNode* i, DomNode* j) {
    while (i->sid() != j->sid()) {
        while (i->sid() < j->sid()) j = j->idom_;
        while (j->sid() < i->sid()) i = i->idom_;
    }

    return i;
}

DomNode* DomTree::post_lca(DomNode* i, DomNode* j) {
    while (i->sid() != j->sid()) {
        while (i->sid() > j->sid()) j = j->idom_;
        while (j->sid() > i->sid()) i = i->idom_;
    }

    return i;
}

const DomNode* DomTree::node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[lambda->sid()]; }
DomNode* DomTree::lookup(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[lambda->sid()]; }
size_t DomTree::size() const { return scope_.size(); }

} // namespace anydsl2
