#include "thorin/analyses/domfrontier.h"

#include "thorin/analyses/domtree.h"

namespace thorin {

template<bool forward>
void DomFrontierBase<forward>::create() {
    const auto& domtree = cfg().domtree();
    for (auto n : cfg().reverse_post_order().skip_front()) {
        const auto& preds = cfg().preds(n);
        if (preds.size() > 1) {
            auto idom = domtree.idom(n);
            for (auto pred : preds) {
                for (auto i = pred; i != idom; i = domtree.idom(i))
                    link(i, n);
            }
        }
    }
}

template<bool forward>
void DomFrontierBase<forward>::stream_ycomp(std::ostream& out) const {
    thorin::ycomp(out, forward ? YCompOrientation::TopToBottom : YCompOrientation::BottomToTop,
                  scope(), make_range(cfg().reverse_post_order()),
                  [&] (const CFNode* n) { return make_range(succs(n)); });
}

template class DomFrontierBase<true>;
template class DomFrontierBase<false>;

}
