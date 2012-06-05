#ifndef ANYDSL_DUMP_H
#define ANYDSL_DUMP_H

#include <ostream>

namespace anydsl {

class AIRNode;

void dump(const AIRNode* n, std::ostream& s = std::cout);

} // namespace anydsl

#endif
