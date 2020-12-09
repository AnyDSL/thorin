#include "thorin/set.h"

#include "thorin/lam.h"

namespace thorin {

template<bool up>
size_t Bound<up>::find(const Def* type) const {
    auto i = isa_nominal()
        ? std::  find(ops().begin(), ops().end(), type)
        : binary_find(ops().begin(), ops().end(), type, GIDLt<const Def*>());
    return i == ops().end() ? size_t(-1) : i - ops().begin();
}

template size_t Bound<false>::find(const Def*) const;
template size_t Bound<true >::find(const Def*) const;

}
