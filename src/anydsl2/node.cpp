#include "anydsl2/node.h"

#include <typeinfo>

namespace anydsl2 {

size_t Node::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, kind());
    boost::hash_combine(seed, size());
    for_all (op, ops_)
        boost::hash_combine(seed, op);

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
