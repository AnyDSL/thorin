#include "thorin/analyses/domtree.h"

#include <limits>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool forwards>
int DomNodeBase<forwards>::depth() const {
    int result = 0;
    for (const DomNodeBase* i = this; !i->entry(); i = i->idom())
        ++result;
    return result;
};

//------------------------------------------------------------------------------

template<bool forwards>
void DomTreeBase<forwards>::create() {
    for (auto lambda : Super::rpo())
        Super::nodes_[lambda] = new DomNode(lambda);

    // map entries' initial idoms to themselves
    for (auto entry : Super::entries()) {
        DomNode* entry_node = Super::lookup(entry);
        entry_node->idom_ = entry_node;
    }

    // all others' idoms are set to their first found dominating pred
    for (auto lambda : Super::body()) {
        for (auto pred : Super::preds(lambda)) {
            if (Super::sid(pred) < Super::sid(lambda)) {
                Super::lookup(lambda)->idom_ = Super::lookup(pred);
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for (auto lambda : Super::body()) {
            DomNode* lambda_node = Super::lookup(lambda);

            DomNode* new_idom = 0;
            for (auto pred : Super::preds(lambda)) {
                DomNode* pred_node = Super::lookup(pred);
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

    for (auto lambda : Super::body()) {
        const DomNode* n = Super::lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

template<bool forwards>
DomNodeBase<forwards>* DomTreeBase<forwards>::lca(DomNode* i, DomNode* j) {
    while (!Super::is_entry(i, j) && Super::sid(i) != Super::sid(j)) {
        while (!Super::is_entry(i, j) && Super::sid(i) < Super::sid(j)) 
            j = j->idom_;
        while (!Super::is_entry(i, j) && Super::sid(j) < Super::sid(i)) 
            i = i->idom_;
    }

    return i;
}

// export templates
template class DomNodeBase< true>;
template class DomNodeBase<false>;
template class DomTreeBase< true>;
template class DomTreeBase<false>;

} // namespace thorin
