#include "thorin/analyses/domtree.h"

#include <iostream>
#include <limits>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"

namespace thorin {

//------------------------------------------------------------------------------

int DomNode::depth() const {
    int result = 0;
    for (const DomNode* i = this; !i->entry(); i = i->idom())
        ++result;
    return result;
};

//------------------------------------------------------------------------------

DomTree::DomTree(const Scope& scope)
    : scope_(scope)
    , is_forward_(scope.is_forward())
{
    create();
}

void DomTree::create() {
    for (auto lambda : scope())
        map_[lambda] = new DomNode(lambda);

    // map entry's initial idom to itself
    root_ = lookup(scope().entry());
    root_->idom_ = root_;

    // all others' idom are set to their first found dominating pred
    for (auto lambda : scope().body()) {
        for (auto pred : scope().preds(lambda)) {
            assert(scope().contains(pred));
            if (scope().sid(pred) < scope().sid(lambda)) {
                auto n = lookup(pred);
                assert(n);
                lookup(lambda)->idom_ = n;
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for (auto lambda : scope().body()) {
            DomNode* lambda_node = lookup(lambda);

            DomNode* new_idom = nullptr;
            for (auto pred : scope().preds(lambda)) {
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

    for (auto lambda : scope().body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

DomNode* DomTree::lca(DomNode* i, DomNode* j) {
    assert(scope().is_forward() == is_forward());
    assert(i && j);
    auto sid = [&] (DomNode* n) { return scope().sid(n->lambda()); };

    while (sid(i) != sid(j)) {
        while (sid(i) < sid(j)) j = j->idom_;
        while (sid(j) < sid(i)) i = i->idom_;
    }

    return i;
}

//------------------------------------------------------------------------------

void DomNode::dump() const {
    for (int i = 0, e = depth(); i != e; ++i)
        std::cout << '\t';
    std::cout << lambda()->unique_name() << std::endl;
    for (auto child : children())
        child->dump();
}

}
