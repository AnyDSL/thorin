#include "thorin/analyses/domtree.h"

namespace thorin {

template<bool forward>
void DomTreeBase<forward>::create() {
    // Cooper et al, 2001. A Simple, Fast Dominance Algorithm. http://www.cs.rice.edu/~keith/EMBED/dom.pdf

    // all idoms different from entry are set to their first found dominating pred
    for (auto n : cfg().reverse_post_order().skip_front()) {
        for (auto pred : cfg().preds(n)) {
            if (cfg().index(pred) < cfg().index(n)) {
                idoms_[n] = pred;
                goto outer_loop;
            }
        }
        THORIN_UNREACHABLE;
outer_loop:;
    }

    for (bool todo = true; todo;) {
        todo = false;

        for (auto n : cfg().reverse_post_order().skip_front()) {
            const CFNode* new_idom = nullptr;
            for (auto pred : cfg().preds(n))
                new_idom = new_idom ? least_common_ancestor(new_idom, pred) : pred;

            assert(new_idom);
            if (idom(n) != new_idom) {
                idoms_[n] = new_idom;
                todo = true;
            }
        }
    }

    for (auto n : cfg().reverse_post_order().skip_front())
        children_[idom(n)].push_back(n);
}

template<bool forward>
void DomTreeBase<forward>::depth(const CFNode* n, int i) {
    depth_[n] = i;
    for (auto child : children(n))
        depth(child, i+1);
}

template<bool forward>
const CFNode* DomTreeBase<forward>::least_common_ancestor(const CFNode* i, const CFNode* j) const {
    assert(i && j);
    while (index(i) != index(j)) {
        while (index(i) < index(j)) j = idom(j);
        while (index(j) < index(i)) i = idom(i);
    }
    return i;
}

template<bool forward>
void DomTreeBase<forward>::stream_ycomp(std::ostream& out) const {
    thorin::ycomp(out, YCompOrientation::TopToBottom, scope(), range(cfg().reverse_post_order()),
        [&] (const CFNode* n) { return range(children(n)); }
    );
}

template class DomTreeBase<true>;
template class DomTreeBase<false>;

}
