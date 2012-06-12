#include "impala/ast.h"

#include "anydsl/util/cast.h"

using anydsl::dcast;

namespace impala {

Decl::Decl(const Token& tok, const Type* type)
    : symbol_(tok.symbol())
    , type_(type)
{
    loc = anydsl::Location(tok.pos1(), type->loc.pos2());
}

void Fct::set(const anydsl::Position& pos1, const anydsl::Symbol symbol, const Type* retType, const Stmt* body) {
    symbol_ = symbol;
    retType_ = retType;
    body_ = body;
    loc = anydsl::Location(pos1, body->loc.pos2());
}

/*
 * types
 */

PrimType::PrimType(const anydsl::Location& loc, Kind kind)
    : kind_(kind)
{
    this->loc = loc;
}

bool PrimType::equal(const Type* t) const {
    if (const PrimType* p = dcast<PrimType>(t)) {
        return kind() == p->kind();
    }

    return false;
}

const PrimType* PrimType::clone(const anydsl::Location& loc) const {
    return new PrimType(loc, kind());
}

/*
 * Expr
 */

Literal::Literal(const anydsl::Location& loc, Kind kind, anydsl::Box value)
    : kind_(kind)
    , value_(value)
{
    this->loc = loc;
}

Id::Id(const Token& tok) 
    : symbol_(tok.symbol())
{
    loc = tok.loc();
}

PrefixExpr::PrefixExpr(const anydsl::Position& pos1, Kind kind, const Expr* right)
    : kind_(kind)
{
    args_.push_back(right);
    loc = anydsl::Location(pos1, right->loc.pos2());
}

InfixExpr::InfixExpr(const Expr* left, Kind kind, const Expr* right)
    : kind_(kind)
{
    args_.push_back(left);
    args_.push_back(right);
    loc = anydsl::Location(left->loc.pos1(), right->loc.pos2());
}

PostfixExpr::PostfixExpr(const Expr* left, Kind kind, const anydsl::Position& pos2) 
    : kind_(kind)
{
    args_.push_back(left);
    loc = anydsl::Location(left->loc.pos1(), pos2);
}

/*
 * Stmt
 */

ExprStmt::ExprStmt(const Expr* expr, const anydsl::Position& pos2)
    : expr_(expr)
{
    loc = anydsl::Location(expr->loc.pos1(), pos2);
}

DeclStmt::DeclStmt(const Decl* decl, const Expr* init, const anydsl::Position& pos2)
    : decl_(decl)
    , init_(init)
{
    loc = anydsl::Location(decl->loc.pos1(), pos2);
}

IfElseStmt::IfElseStmt(const anydsl::Position& pos1, const Expr* cond, const Stmt* ifStmt, const Stmt* elseStmt)
    : cond_(cond)
    , ifStmt_(ifStmt)
    , elseStmt_(elseStmt)
{
    loc = anydsl::Location(pos1, elseStmt->loc.pos2());
}

void WhileStmt::set(const anydsl::Position& pos1, const Expr* cond, const Stmt* body) {
    Loop::set(cond, body);
    loc = anydsl::Location(pos1, body->loc.pos2());
}

void DoWhileStmt::set(const anydsl::Position& pos1, const Stmt* body, const Expr* cond, const anydsl::Position& pos2) {
    Loop::set(cond, body);
    loc = anydsl::Location(pos1, pos2);
}

ForStmt::~ForStmt() {
    if (isDecl())
        delete initDecl_;
    else
        delete initExpr_;
}

void ForStmt::set(const anydsl::Position& pos1, const Expr* cond, const Expr* inc, const Stmt* body) {
    Loop::set(cond, body);
    inc_ = inc;
    loc = anydsl::Location(pos1, body->loc.pos2());
}


BreakStmt::BreakStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    loc = anydsl::Location(pos1, pos2);
}

ContinueStmt::ContinueStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    loc = anydsl::Location(pos1, pos2);
}

ReturnStmt::ReturnStmt(const anydsl::Position& pos1, const Expr* expr, const anydsl::Position& pos2)
    : expr_(expr)
{
    loc = anydsl::Location(pos1, pos2);
}

} // namespace impala
