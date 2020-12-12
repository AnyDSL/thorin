#include "thorin/lattice.h"

#include "thorin/lam.h"
#include "thorin/world.h"

namespace thorin {

size_t Bound::find(const Def* type) const {
    auto i = isa_nominal()
        ? std::  find(ops().begin(), ops().end(), type)
        : binary_find(ops().begin(), ops().end(), type, GIDLt<const Def*>());
    return i == ops().end() ? size_t(-1) : i - ops().begin();
}

const Lit* Bound::index(const Def* type) const { return world().lit_int(num_ops(), find(type)); }
const Sigma* Bound::convert() const { return isa<Join>() ? as<Join>()->convert() : as<Meet>()->convert(); }

template<bool up>
const Sigma* TBound<up>::convert() const {
    auto& w = world();

    if constexpr (up) {
        nat_t align = 0;
        nat_t size  = 0;

        for (auto op : ops()) {
            auto a = isa_lit(w.op(Trait::align, op));
            auto s = isa_lit(w.op(Trait::size , op));
            if (!a || !s) return nullptr;

            align = std::max(align, *a);
            size  = std::max(size , *s);
        }

        assert(size % align == 0);
        auto arr = w.arr(size / align, w.type_int_width(align));

        return w.sigma({w.type_int(num_ops()), arr})->as<Sigma>();
    } else {
        return w.sigma(ops())->as<Sigma>();
    }
}

template const Sigma* TBound<false>::convert() const;
template const Sigma* TBound<true >::convert() const;

}
