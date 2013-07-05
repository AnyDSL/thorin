#include "anydsl2/analyses/domtree.h"

#include <limits>
#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

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
    t_for_all (lambda, Super::rpo())
        Super::nodes_[Super::sid(lambda)] = new DomNode(lambda);

    for (size_t i = 0; i < Super::size(); ++i)
        assert(i == Super::sid(Super::nodes_[i]));

    // map entries' initial idoms to themselves
    t_for_all (entry,  Super::entries()) {
        DomNode* entry_node = Super::lookup(entry);
        entry_node->idom_ = entry_node;
    }

    // all others' idoms are set to their first found dominating pred
    t_for_all (lambda, Super::body()) {
        t_for_all (pred, Super::preds(lambda)) {
            if (Super::sid(pred) < Super::sid(lambda)) {
                Super::lookup(lambda)->idom_ = Super::lookup(pred);
                goto outer_loop;
            }
        }
        ANYDSL2_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        t_for_all (lambda, Super::body()) {
            DomNode* lambda_node = Super::lookup(lambda);

            DomNode* new_idom = 0;
            t_for_all (pred, Super::preds(lambda)) {
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

    t_for_all (lambda, Super::body()) {
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

} // namespace anydsl2
