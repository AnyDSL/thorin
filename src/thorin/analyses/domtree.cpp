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

Lambda* DomNode::lambda() const { return cfg_node()->lambda(); }

//------------------------------------------------------------------------------

template<bool forward>
DomTreeBase<forward>::DomTreeBase(const CFGView<forward>& cfg)
    : cfg_(cfg)
    , nodes_(cfg.size())
{
    create();
}

template<bool forward>
void DomTreeBase<forward>::create() {
    for (auto n : cfg())
        _lookup(n) = new DomNode(n);

    // map entry's initial idom to itself
    root_ = _lookup(cfg().entry());
    root_->idom_ = root_;

    // all others' idom are set to their first found dominating pred
    for (auto n : cfg().body()) {
        for (auto pred : cfg().preds(n)) {
            if (cfg().rpo_id(pred) < cfg().rpo_id(n)) {
                auto dom = _lookup(pred);
                assert(dom);
                _lookup(n)->idom_ = dom;
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool changed = true; changed;) {
        changed = false;

        for (auto n : cfg().body()) {
            auto dom = _lookup(n);

            DomNode* new_idom = nullptr;
            for (auto pred : cfg().preds(n)) {
                auto pred_dom = _lookup(pred);
                assert(pred_dom);
                new_idom = new_idom ? _lca(new_idom, pred_dom) : pred_dom;
            }
            assert(new_idom);
            if (dom->idom() != new_idom) {
                dom->idom_ = new_idom;
                changed = true;
            }
        }
    }

    for (auto n : cfg().body()) {
        auto dom = _lookup(n);
        dom->idom_->children_.push_back(dom);
    }

    auto num = postprocess(root_, 0);
    assert(num = cfg().size());
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
DomNode* DomTreeBase<forward>::_lca(DomNode* i, DomNode* j) {
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
