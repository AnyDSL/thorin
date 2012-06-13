#include "impala/init.h"

#include "anydsl/symbol.h"

#include "impala/prec.h"
#include "impala/token.h"

namespace impala {

void init() {
    PrecTable::init();
    Token::init();
}

void destroy() {
    anydsl::Symbol::destroy();
}

} // namespace impala

