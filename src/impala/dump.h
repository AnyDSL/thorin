#ifndef IMPALA_DUMP_H
#define IMPALA_DUMP_H

#include <iostream>

namespace impala {

/** 
 * @brief Dumps a human readable representation of the ASTNode \p n to output stream \p o.
 * 
 * @param fancy If set, the dumper makes minimal use of parenthesis for expressions.
 *  Otherwise, everything is put in parenthesis in order to fully debug the internal tree structure.
 */
void dump(const ASTNode* n, bool fancy = false, std::ostream& o = std::cout);

} // namespace impala

#endif // IMPALA_DUMP_H
