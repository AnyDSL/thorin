#include "anydsl2/node.h"

namespace anydsl2 {

size_t Node::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, kind());
    hash_combine(seed, ops_);
    return seed;
}

bool Node::equal(const Node* other) const {
    return this->kind() == other->kind() && this->ops_ == other->ops_;
}

} // namespace anydsl2
