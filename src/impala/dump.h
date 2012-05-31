#ifndef IMPALA_DUMP_H
#define IMPALA_DUMP_H

#include <iostream>

namespace impala {

void dump(const ASTNode* n, bool fancy = false, std::ostream& o = std::cout);

} // namespace impala

#endif // IMPALA_DUMP_H
