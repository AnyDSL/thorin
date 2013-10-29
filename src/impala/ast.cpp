#include "impala/ast.h"

#include "impala/type.h"

using namespace anydsl2;

namespace impala {

TokenKind Literal::literal2type() const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: return Token::TYPE_##itype;
#include "impala/tokenlist.h"
        case LIT_bool:    return Token::TYPE_bool;
        default: ANYDSL2_UNREACHABLE;
    }
}

bool IndexExpr::is_array_subscript() const {
    return lhs()->type()->isa<ArrayType>() != nullptr;
}

bool Call::is_continuation_call() const { return type()->isa<NoRet>() != nullptr; }

Location Call::args_location() const {
    if (ops().size() == 1)
        return Location(pos2());
    return Location(op(1)->pos1(), ops_.back()->pos2());
}

bool ForStmt::is_while() const { 
    if (const ExprStmt* expr_stmt = init()->isa<ExprStmt>())
        return expr_stmt->expr()->isa<EmptyExpr>() && step()->isa<EmptyExpr>();
    return false;
}

uint64_t Literal::get_u64() const { return bcast<uint64_t, Box>(box()); }

} // namespace impala
