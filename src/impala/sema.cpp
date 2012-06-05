#include "impala/ast.h"

#include "anydsl/util/for_all.h"

namespace impala {

//------------------------------------------------------------------------------

class Sema {
    Sema()
        : result(true)
    {}

    bool result;
};

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) const {
    for_all (f, fcts())
        f->check(sema);
}

void Fct::check(Sema& sema) const {
    for_all (p, params())
        p->check(sema);

    if (retType())
        retType()->check(sema);

    body()->check(sema);
}

void Decl::check(Sema& sema) const {
    type()->check(sema);
}

/*
 * Type
 */

void PrimType::check(Sema& sema) const {
    /* do nothing */
}


/*
 * Expr
 */

void EmptyExpr::check(Sema& sema) const {
    /* do nothing */
}

void Literal::check(Sema& sema) const {
    /* do nothing */
}

void Id::check(Sema& sema) const {
}

void PrefixExpr::check(Sema& sema) const {
}

void InfixExpr::check(Sema& sema) const {
}

void PostfixExpr::check(Sema& sema) const {
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const {
}

void ExprStmt::check(Sema& sema) const {
}

void IfElseStmt::check(Sema& sema) const {
}

void WhileStmt::check(Sema& sema) const {
}

void DoWhileStmt::check(Sema& sema) const {
}

void ForStmt::check(Sema& sema) const {
}

void BreakStmt::check(Sema& sema) const {
}

void ContinueStmt::check(Sema& sema) const {
}

void ReturnStmt::check(Sema& sema) const {
}

void ScopeStmt::check(Sema& sema) const {
}

} // namespace impala
