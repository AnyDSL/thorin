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
DomTreeBase<forwards>::~DomTreeBase() {
    t_for_all (node, nodes_)
        delete node;
}

template<bool forwards>
void DomTreeBase<forwards>::create() {
    t_for_all (lambda, rpo())
        nodes_[index(lambda)] = new DomNode(lambda);

    for (size_t i = 0; i < size(); ++i)
        assert(i == index(nodes_[i]));

    // map entries' initial idoms to themselves
    t_for_all (entry,  entries()) {
        DomNode* entry_node = lookup(entry);
        entry_node->idom_ = entry_node;
    }

    // all others' idoms are set to their first found dominating pred
    t_for_all (lambda, body()) {
        t_for_all (pred, preds(lambda)) {
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

        t_for_all (lambda, body()) {
            DomNode* lambda_node = lookup(lambda);

            DomNode* new_idom = 0;
            t_for_all (pred, preds(lambda)) {
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

    t_for_all (lambda, body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

template<bool forwards>
DomNodeBase<forwards>* DomTreeBase<forwards>::lca(DomNode* i, DomNode* j) {
    while (!is_entry(i, j) && index(i) != index(j)) {
        while (!is_entry(i, j) && index(i) < index(j)) 
            j = j->idom_;
        while (!is_entry(i, j) && index(j) < index(i)) 
            i = i->idom_;
    }

    return i;
}

template<bool forwards> size_t DomTreeBase<forwards>::size() const { return scope_.size(); }

// export templates
template class DomNodeBase< true>; template class DomTreeBase< true>;   
template class DomNodeBase<false>; template class DomTreeBase<false>;  

} // namespace anydsl2
