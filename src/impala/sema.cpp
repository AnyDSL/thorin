#include "impala/ast.h"

namespace impala {

//------------------------------------------------------------------------------

class Sema {
    Sema()
        : result(true)
    {}

    bool result;
};

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) {
}

void Fct::check(Sema& sema) {
}

void Decl::check(Sema& sema) {
}

/*
 * Type
 */

void PrimType::check(Sema& sema) {
}


/*
 * Expr
 */

void EmptyExpr::check(Sema& sema) {
}

void Literal::check(Sema& sema) {
}

void Id::check(Sema& sema) {
}

void PrefixExpr::check(Sema& sema) {
}

void InfixExpr::check(Sema& sema) {
}

void PostfixExpr::check(Sema& sema) {
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) {
}

void ExprStmt::check(Sema& sema) {
}

void IfElseStmt::check(Sema& sema) {
}

void WhileStmt::check(Sema& sema) {
}

void DoWhileStmt::check(Sema& sema) {
}

void ForStmt::check(Sema& sema) {
}

void BreakStmt::check(Sema& sema) {
}

void ContinueStmt::check(Sema& sema) {
}

void ReturnStmt::check(Sema& sema) {
}

void ScopeStmt::check(Sema& sema) {
}

} // namespace impala
