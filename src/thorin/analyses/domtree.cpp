#include "thorin/analyses/domtree.h"

#include <iostream>
#include <limits>
#include <queue>

#include "thorin/lambda.h"
#include "thorin/analyses/cfg.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool forward>
void DomTreeBase<forward>::Node::dump(const int depth) const {
    for (int i = 0, e = depth; i != e; ++i)
        std::cout << '\t';
    std::cout << cf_node()->def()->unique_name() << std::endl;
    for (auto child : children())
        child->dump(depth+1);
}

//------------------------------------------------------------------------------

template<bool forward>
void DomTreeBase<forward>::create() {
    for (auto n : cfg().rpo())
        nodes_[n] = new Node(n);

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

            const Node* new_idom = nullptr;
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

    assert((*this)[cfg().entry()] == root());
}

template<bool forward>
const typename DomTreeBase<forward>::Node* DomTreeBase<forward>::lca(const Node* i, const Node* j) const {
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
