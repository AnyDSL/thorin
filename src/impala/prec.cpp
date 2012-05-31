#include "impala/prec.h"

namespace impala {

/*static*/ Type2Prec PrecTable::prefix_r;
/*static*/ Type2Prec PrecTable::infix_l;
/*static*/ Type2Prec PrecTable::infix_r;
/*static*/ Type2Prec PrecTable::postfix_l;
/*static*/ PrecTable::ForceInit PrecTable::init_;

PrecTable::ForceInit::ForceInit() {
#define IMPALA_PREFIX(    tok, t_str,    r)  PrecTable::prefix_r[Token:: tok] = r;
#define IMPALA_POSTFIX(   tok, t_str, l   ) PrecTable::postfix_l[Token:: tok] = l;
#define IMPALA_INFIX(     tok, t_str, l, r)   PrecTable::infix_l[Token:: tok] = l; PrecTable::infix_r[Token:: tok] = r;
#define IMPALA_INFIX_ASGN(tok, t_str, l, r)   PrecTable::infix_l[Token:: tok] = l; PrecTable::infix_r[Token:: tok] = r;
#include <impala/tokenlist.h>
}

} // namespace impala
