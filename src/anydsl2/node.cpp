#include "anydsl2/node.h"

#include <typeinfo>

#include "anydsl2/util/hash.h"

namespace anydsl2 {

size_t Node::hash() const {
    size_t seed = hash_combine(hash_value(kind()), size());
    for (auto op : ops_)
        seed = hash_combine(seed, op);
    return seed;
}

bool Node::equal(const Node* other) const {
    if (typeid(*this) == typeid(*other) && this->size() == other->size()) {
        for_all2 (this_op, this->ops_, other_op, other->ops_) {
            if (this_op != other_op)
                return false;
        }
        return true;
    }
    return false;
}

}
