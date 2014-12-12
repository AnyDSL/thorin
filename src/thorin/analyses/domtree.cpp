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
DomTreeBase<forward>::DomTreeBase(const CFG<forward>& cfg)
    : cfg_(cfg)
    , nodes_(cfg)
{
    create();
}

template<bool forward>
void DomTreeBase<forward>::create() {
    for (auto n : cfg())
        nodes_[n] = new DomNode(n);

    // map entry's initial idom to itself
    root_ = nodes_[cfg().entry()];
    root_->idom_ = root_;

    // all others' idom are set to their first found dominating pred
    for (auto n : cfg().body()) {
        for (auto pred : cfg().preds(n)) {
            if (cfg().index(pred) < cfg().index(n)) {
                auto dom = nodes_[pred];
                assert(dom);
                nodes_[n]->idom_ = dom;
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool todo = true; todo;) {
        todo = false;

        for (auto n : cfg().body()) {
            auto dom = nodes_[n];

            const DomNode* new_idom = nullptr;
            for (auto pred : cfg().preds(n)) {
                auto pred_dom = nodes_[pred];
                assert(pred_dom);
                new_idom = new_idom ? lca(new_idom, pred_dom) : pred_dom;
            }
            assert(new_idom);
            if (dom->idom() != new_idom) {
                dom->idom_ = new_idom;
                todo = true;
            }
        }
    }

    for (auto n : cfg().body()) {
        auto dom = nodes_[n];
        dom->idom_->children_.push_back(dom);
    }

    auto num = postprocess(root_, 0);
    assert(num = cfg().size());
}

template<bool forward>
size_t DomTreeBase<forward>::postprocess(const DomNode* n, int depth) {
    n->depth_ = depth;
    n->max_index_ = 0;
    for (auto child : n->children())
        n->max_index_ = std::max(n->max_index_, postprocess(child, depth+1));
    return n->max_index_;
}

template<bool forward>
const DomNode* DomTreeBase<forward>::lca(const DomNode* i, const DomNode* j) const {
    assert(i && j);
    while (index(i) != index(j)) {
        while (index(i) < index(j)) j = j->idom_;
        while (index(j) < index(i)) i = i->idom_;
    }
    return i;
}

template class DomTreeBase<true>;
template class DomTreeBase<false>;

//------------------------------------------------------------------------------

}
