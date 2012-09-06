#ifndef ANYDSL_ANALYSES_PLACEMENT_H
#define ANYDSL_ANALYSES_PLACEMENT_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class DomNode;

void place(const DomNode* root);

} // namespace anydsl

#endif
