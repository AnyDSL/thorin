#include "thorin/analyses/domtree.h"

#include <iostream>
#include <limits>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/analyses/cfg.h"

namespace thorin {

//------------------------------------------------------------------------------

void DomNode::dump() const {
    for (int i = 0, e = depth(); i != e; ++i)
        std::cout << '\t';
    std::cout << cfg_node()->lambda()->unique_name() << std::endl;
    for (auto child : children())
        child->dump();
}

//------------------------------------------------------------------------------

template<bool forward>
DomTreeBase<forward>::DomTreeBase(const CFGView<forward>& cfg_view)
    : cfg_view_(cfg_view)
    , nodes_(cfg_view.size())
{
    create();
}

template<bool forward>
void DomTreeBase<forward>::create() {
    for (auto n : cfg_view())
        lookup(n) = new DomNode(n);

    // map entry's initial idom to itself
    root_ = lookup(cfg_view().entry());
    root_->idom_ = root_;

    // all others' idom are set to their first found dominating pred
    for (auto lambda : cfg_view().body()) {
        for (auto pred : cfg_view().preds(lambda)) {
            if (cfg_view().rpo_id(pred) < cfg_view().rpo_id(lambda)) {
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

        for (auto lambda : cfg_view().body()) {
            DomNode* lambda_node = lookup(lambda);

            DomNode* new_idom = nullptr;
            for (auto pred : cfg_view().preds(lambda)) {
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

    for (auto lambda : cfg_view().body()) {
        auto n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }

    auto n = postprocess(root_, 0);
    assert(n = cfg_view().size());
}

template<bool forward>
size_t DomTreeBase<forward>::postprocess(DomNode* n, int depth) {
    n->depth_ = depth;
    n->max_rpo_id_ = 0;
    for (auto child : n->children())
        n->max_rpo_id_ = std::max(n->max_rpo_id_, postprocess(const_cast<DomNode*>(child), depth+1));
    return n->max_rpo_id_;
}

template<bool forward>
DomNode* DomTreeBase<forward>::lca(DomNode* i, DomNode* j) {
    assert(i && j);
    while (rpo_id(i) != rpo_id(j)) {
        while (rpo_id(i) < rpo_id(j)) j = j->idom_;
        while (rpo_id(j) < rpo_id(i)) i = i->idom_;
    }
    return i;
}

template class DomTreeBase<true>;
template class DomTreeBase<false>;

//------------------------------------------------------------------------------

}
