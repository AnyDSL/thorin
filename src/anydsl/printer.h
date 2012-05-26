#ifndef ANYDSL_PRINTER_H
#define ANYDSL_PRINTER_H

#include <ostream>

namespace anydsl {

class AIRNode;

void print(std::ostream& s, const AIRNode* n);

} // namespace anydsl

#endif
