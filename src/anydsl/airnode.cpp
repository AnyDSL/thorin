#include "anydsl/airnode.h"

#include "anydsl/dump.h"

namespace anydsl {

void AIRNode::dump() {
    ::anydsl::dump(this);
}

} // namespace anydsl
