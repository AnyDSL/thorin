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
        for (size_t i = 0, e = size(); i != e; ++i) {
            if (this->ops_[i] != other->ops_[i])
                return false;
        }
        return true;
    }
    return false;
}

}
