#include "impala/ast.h"

#include "anydsl/cfg.h"
#include "anydsl/util/cast.h"

#include "impala/dump.h"

using anydsl::dcast;
using anydsl::Location;
using anydsl::Position;
using anydsl::Symbol;

namespace impala {

void ASTNode::dump() const {
    ::impala::dump(this);
}

Decl::Decl(const Token& tok, const Type* type, const Position& pos2)
    : symbol_(tok.symbol())
    , type_(type)
{
    setLoc(tok.pos1(), pos2);
}

void Fct::set(const Position& pos1, const Symbol symbol, const Type* retType, const ScopeStmt* body) {
    symbol_ = symbol;
    retType_ = retType;
    body_ = body;
    setLoc(pos1, body->pos2());
}

/*
 * Expr
 */

Literal::Literal(const Location& loc, Kind kind, anydsl::Box value)
    : kind_(kind)
    , value_(value)
{
    loc_= loc;
}

Id::Id(const Token& tok) 
    : symbol_(tok.symbol())
{
    loc_ = tok.loc();
}

PrefixExpr::PrefixExpr(const Position& pos1, Kind kind, const Expr* right)
    : kind_(kind)
{
    args_.push_back(right);
    setLoc(pos1, right->pos2());
}

InfixExpr::InfixExpr(const Expr* left, Kind kind, const Expr* right)
    : kind_(kind)
{
    args_.push_back(left);
    args_.push_back(right);
    setLoc(left->pos1(), right->pos2());
}

PostfixExpr::PostfixExpr(const Expr* left, Kind kind, const Position& pos2) 
    : kind_(kind)
{
    args_.push_back(left);
    setLoc(left->pos1(), pos2);
}

/*
 * Stmt
 */

ExprStmt::ExprStmt(const Expr* expr, const Position& pos2)
    : expr_(expr)
{
    setLoc(expr->pos1(), pos2);
}

DeclStmt::DeclStmt(const Decl* decl, const Expr* init, const Position& pos2)
    : decl_(decl)
    , init_(init)
{
    setLoc(decl->pos1(), pos2);
}

IfElseStmt::IfElseStmt(const Position& pos1, const Expr* cond, const Stmt* thenStmt, const Stmt* elseStmt)
    : cond_(cond)
    , thenStmt_(thenStmt)
    , elseStmt_(elseStmt)
{
    setLoc(pos1, elseStmt->pos2());
}

void WhileStmt::set(const Position& pos1, const Expr* cond, const Stmt* body) {
    Loop::set(cond, body);
    setLoc(pos1, body->pos2());
}

void DoWhileStmt::set(const Position& pos1, const Stmt* body, const Expr* cond, const Position& pos2) {
    Loop::set(cond, body);
    setLoc(pos1, pos2);
}

ForStmt::~ForStmt() {
    if (isDecl())
        delete initDecl_;
    else
        delete initExpr_;
}

void ForStmt::set(const Position& pos1, const Expr* cond, const Expr* inc, const Stmt* body) {
    Loop::set(cond, body);
    inc_ = inc;
    setLoc(pos1, body->pos2());
}


BreakStmt::BreakStmt(const Position& pos1, const Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    setLoc(pos1, pos2);
}

ContinueStmt::ContinueStmt(const Position& pos1, const Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    setLoc(pos1, pos2);
}

ReturnStmt::ReturnStmt(const Position& pos1, const Expr* expr, const Fct* fct, const Position& pos2)
    : expr_(expr)
    , fct_(fct)
{
    setLoc(pos1, pos2);
}

} // namespace impala
