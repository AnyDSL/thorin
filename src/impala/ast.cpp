#include "impala/ast.h"

#include "anydsl/util/cast.h"

#include "impala/dump.h"

using anydsl::dcast;
using anydsl::Location;

namespace impala {

void ASTNode::dump() const {
    ::impala::dump(this);
}

Decl::Decl(const Token& tok, const Type* type)
    : symbol_(tok.symbol())
    , type_(type)
{
    setLoc(tok.pos1(), type->pos2());
}

void Fct::set(const anydsl::Position& pos1, const anydsl::Symbol symbol, const Type* retType, const ScopeStmt* body) {
    symbol_ = symbol;
    retType_ = retType;
    body_ = body;
    setLoc(pos1, body->pos2());
}

/*
 * types
 */

PrimType::PrimType(const Location& loc, Kind kind)
    : kind_(kind)
{
    loc_ = loc;
}

bool PrimType::equal(const Type* t) const {
    if (const PrimType* p = dcast<PrimType>(t)) {
        return kind() == p->kind();
    }

    return false;
}

const PrimType* PrimType::clone(const Location& loc) const {
    return new PrimType(loc, kind());
}

Void::Void(const Location& loc) {
    loc_ = loc;
}

bool Void::equal(const Type* t) const {
    return dcast<Void>(t);
}

const Void* Void::clone(const Location& loc) const {
    return new Void(loc);
}

ErrorType::ErrorType(const Location& loc) {
    loc_ = loc;
}

bool ErrorType::equal(const Type* t) const {
    return true;
}

const ErrorType* ErrorType::clone(const Location& loc) const {
    return new ErrorType(loc);
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

PrefixExpr::PrefixExpr(const anydsl::Position& pos1, Kind kind, const Expr* right)
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

PostfixExpr::PostfixExpr(const Expr* left, Kind kind, const anydsl::Position& pos2) 
    : kind_(kind)
{
    args_.push_back(left);
    setLoc(left->pos1(), pos2);
}

/*
 * Stmt
 */

ExprStmt::ExprStmt(const Expr* expr, const anydsl::Position& pos2)
    : expr_(expr)
{
    setLoc(expr->pos1(), pos2);
}

DeclStmt::DeclStmt(const Decl* decl, const Expr* init, const anydsl::Position& pos2)
    : decl_(decl)
    , init_(init)
{
    setLoc(decl->pos1(), pos2);
}

IfElseStmt::IfElseStmt(const anydsl::Position& pos1, const Expr* cond, const Stmt* ifStmt, const Stmt* elseStmt)
    : cond_(cond)
    , ifStmt_(ifStmt)
    , elseStmt_(elseStmt)
{
    setLoc(pos1, elseStmt->pos2());
}

void WhileStmt::set(const anydsl::Position& pos1, const Expr* cond, const Stmt* body) {
    Loop::set(cond, body);
    setLoc(pos1, body->pos2());
}

void DoWhileStmt::set(const anydsl::Position& pos1, const Stmt* body, const Expr* cond, const anydsl::Position& pos2) {
    Loop::set(cond, body);
    setLoc(pos1, pos2);
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
    setLoc(pos1, body->pos2());
}


BreakStmt::BreakStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    setLoc(pos1, pos2);
}

ContinueStmt::ContinueStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    setLoc(pos1, pos2);
}

ReturnStmt::ReturnStmt(const anydsl::Position& pos1, const Expr* expr, const Fct* fct, const anydsl::Position& pos2)
    : expr_(expr)
    , fct_(fct)
{
    setLoc(pos1, pos2);
}

} // namespace impala
