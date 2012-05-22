#include "anydsl/air/def.h"

#include <typeinfo>

#include "anydsl/air/primop.h"
#include "anydsl/air/type.h"
#include "anydsl/air/use.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

void Def::registerUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) == uses_.end(), "must not be inside the use list");
    uses_.insert(use);
}

void Def::unregisterUse(Use* use) { 
    anydsl_assert(use->def() == this, "use does not point to this def");
    anydsl_assert(uses_.find(use) != uses_.end(), "must be inside the use list");
    uses_.erase(use);
}

World& Def::world() const { 
    return type_->world(); 
}

bool ValueNumber::operator == (const ValueNumber& vn) const {
    if (index != vn.index)
        return false;

    if (hasMore(index)) {
        if (size != vn.size)
            return false;

        bool result = true;
        for (size_t i = 0, e = size; i != e && result; ++i)
            result &= more[i] == vn.more[i];

        return result;
    }

    return op1 == vn.op1 && op2 == vn.op2 && op3 == vn.op3;
}

size_t hash_value(const ValueNumber& vn) {
    size_t seed = 0;

    if (ValueNumber::hasMore(vn.index)) {
        boost::hash_combine(seed, vn.index);
        boost::hash_combine(seed, vn.size);

        for (size_t i = 0, e = vn.size; i != e; ++i)
            boost::hash_combine(seed, vn.more[i]);

        return seed;
    }

    boost::hash_combine(seed, vn.index);
    boost::hash_combine(seed, vn.op1);
    boost::hash_combine(seed, vn.op2);
    boost::hash_combine(seed, vn.op3);

    return seed;
}

} // namespace anydsl
