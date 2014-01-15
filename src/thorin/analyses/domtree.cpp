#include "thorin/analyses/domtree.h"

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

template<bool forwards>
void DomTreeBase<forwards>::create() {
    for (auto lambda : scope_.rpo())
        map_[lambda] = new DomNode(lambda);

    // map entry's initial idom to itself
    DomNode* entry_node = lookup(scope_.entry());
    entry_node->idom_ = entry_node;

    // all others' idoms are set to their first found dominating pred
    for (auto lambda : scope_.body()) {
        for (auto pred : scope_.preds(lambda)) {
            if (scope_.sid(pred) < scope_.sid(lambda)) {
                lookup(lambda)->idom_ = lookup(pred);
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for (auto lambda : scope_.body()) {
            DomNode* lambda_node = lookup(lambda);

            DomNode* new_idom = 0;
            for (auto pred : scope_.preds(lambda)) {
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

    for (auto lambda : scope_.body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

template<bool forwards>
DomNode* DomTreeBase<forwards>::lca(DomNode* i, DomNode* j) {
    auto sid = [&] (DomNode* n) { return scope_.sid(n->lambda()); };

    while (sid(i) != sid(j)) {
        while (sid(i) < sid(j)) j = j->idom_;
        while (sid(j) < sid(i)) i = i->idom_;
    }

    return i;
}

// export templates
template class DomTreeBase< true>;
template class DomTreeBase<false>;

} // namespace thorin
