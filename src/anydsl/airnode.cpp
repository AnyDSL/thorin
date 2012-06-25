#include "anydsl/airnode.h"

#include "anydsl/dump.h"

namespace anydsl {

void AIRNode::dump() const {
    ::anydsl::dump(this);
}

} // namespace anydsl
