#include "impala/init.h"

#include "thorin/util/symbol.h"

#include "impala/prec.h"
#include "impala/token.h"

namespace impala {

void init() { PrecTable::init(); Token::init(); }
void destroy() { thorin::Symbol::destroy(); }

} // namespace impala

