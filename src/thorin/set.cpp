#include "thorin/set.h"

#include "thorin/lam.h"

namespace thorin {

const Lam* Test::match() const { return op(2)->as<Lam>(); }
const Lam* Test::clash() const { return op(3)->as<Lam>(); }

template<bool up>
bool Bound<up>::contains(const Def* type) const {
    if (isa_nominal())
        return std::find(ops().begin(), ops().end(), type);
    return std::binary_search(ops().begin(), ops().end(), type, GIDLt<const Def*>());
}

template bool Bound<false>::contains(const Def*) const;
template bool Bound<true >::contains(const Def*) const;

}
