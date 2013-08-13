#include "impala/init.h"

#include "anydsl2/util/symbol.h"

#include "impala/prec.h"
#include "impala/token.h"

namespace impala {

void init() { PrecTable::init(); Token::init(); }
void destroy() { anydsl2::Symbol::destroy(); }

} // namespace impala

