#include "anydsl/node.h"

namespace anydsl {

size_t Node::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, kind());
    boost::hash_combine(seed, ops_);

    return seed;
}

bool Node::equal(const Node* other) const {
    return this->kind() == other->kind() && this->ops_ == other->ops_;
}

} // namespace anydsl
