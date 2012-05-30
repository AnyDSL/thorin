#include "impala/ast.h"

namespace impala {

std::ostream& Prg::dump(std::ostream& o) {
    return o;
}

std::ostream& Fct::dump(std::ostream& o) {
    return o;
}

std::ostream& Decl::dump(std::ostream& o) {
    return o;
}

/*
 * Type
 */

std::ostream& PrimType::dump(std::ostream& o) {
    return o;
}


/*
 * Expr
 */

std::ostream& EmptyExpr::dump(std::ostream& o) {
    return o;
}

std::ostream& Literal::dump(std::ostream& o) {
    return o;
}

std::ostream& Id::dump(std::ostream& o) {
    return o;
}

std::ostream& PrefixExpr::dump(std::ostream& o) {
    return o;
}

std::ostream& InfixExpr::dump(std::ostream& o) {
    return o;
}

std::ostream& PostfixExpr::dump(std::ostream& o) {
    return o;
}

/*
 * Stmt
 */

std::ostream& EmptyStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& DeclStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& ExprStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& IfElseStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& WhileStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& DoWhileStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& ForStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& BreakStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& ContinueStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& ReturnStmt::dump(std::ostream& o) {
    return o;
}

std::ostream& ScopeStmt::dump(std::ostream& o) {
    return o;
}

} // namespace impala
