#ifndef IMPALA_DUMP_H
#define IMPALA_DUMP_H

#include <iostream>

namespace impala {

class ASTNode;

/** 
 * @brief Dumps a human readable representation of the ASTNode \p n to output stream \p o.
 * 
 * @param n The \p ASTNode to dump.
 * @param fancy If set, the dumper makes minimal use of parenthesis for expressions.
 *  Otherwise, everything is put in parenthesis in order to fully debug the internal tree structure.
 * @param o The output stream where the dump is directed to.
 */
void dump(const ASTNode* n, bool fancy = false, std::ostream& o = std::cout);
void dump(const Type* t, bool fancy = false, std::ostream& o = std::cout);

std::ostream& operator << (std::ostream& o, const ASTNode* n);
std::ostream& operator << (std::ostream& o, const Type* t);

} // namespace impala

#endif // IMPALA_DUMP_H
