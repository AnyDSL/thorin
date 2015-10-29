#include "thorin/analyses/domfrontier.h"

#include "thorin/analyses/domtree.h"

namespace thorin {

template<bool forward>
void DomFrontierBase<forward>::create() {
    const auto& domtree = cfg().domtree();

    // compute the dominance frontier of each node as described in Cooper et al.
    for (const auto n : cfg().body()) {
        const auto& preds = cfg().preds(n);
        if (preds.size() > 1) {
            const auto idom = domtree.idom(n);
            for (const auto pred : preds) {
                auto runner = pred;
                while (runner != idom) {
                    link(n, runner);
                    runner = domtree.idom(runner);
                }
            }
        }
    }
}

template<bool forward>
void DomFrontierBase<forward>::stream_ycomp(std::ostream& out) const {
    thorin::ycomp(out, YCompOrientation::TopToBottom, scope(), range(cfg().rpo()),
        [&] (const CFNode* n) { return range(succs(n)); }
    );
}

template class DomFrontierBase<true>;
template class DomFrontierBase<false>;

}
