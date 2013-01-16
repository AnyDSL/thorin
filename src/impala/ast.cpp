#include "impala/ast.h"

#include "anydsl2/util/cast.h"

#include "impala/type.h"

using anydsl2::Box;
using anydsl2::Location;
using anydsl2::Position;
using anydsl2::Symbol;
using anydsl2::Type;
using anydsl2::dcast;

namespace impala {

//------------------------------------------------------------------------------

Decl::Decl(const Token& tok, const Type* type, const Position& pos2)
    : symbol_(tok.symbol())
    , type_(type)
{
    set_loc(tok.pos1(), pos2);
}

bool Fct::is_continuation() const { return return_type(pi())->isa<NoRet>(); }

void NamedFct::set(const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2) {
    symbol_ = tok.symbol();
    type_ = type;
    set_loc(tok.pos1(), pos2);
}

/*
 * Expr
 */

Literal::Literal(const Location& loc, Kind kind, Box box)
    : kind_(kind)
    , box_(box)
{
    loc_= loc;
}

anydsl2::PrimTypeKind Literal::literal2type() const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: return anydsl2::PrimType_##atype;
#include "impala/tokenlist.h"
        case LIT_bool:    return anydsl2::PrimType_u1;
        default: ANYDSL2_UNREACHABLE;
    }
}

Tuple::Tuple(const Position& pos1) { loc_.set_pos1(pos1); }

Id::Id(const Token& tok) 
    : symbol_(tok.symbol())
{
    loc_ = tok.loc();
}

PrefixExpr::PrefixExpr(const Position& pos1, Kind kind, const Expr* rhs)
    : kind_(kind)
{
    ops_.push_back(rhs);
    set_loc(pos1, rhs->pos2());
}

InfixExpr::InfixExpr(const Expr* lhs, Kind kind, const Expr* rhs)
    : kind_(kind)
{
    ops_.push_back(lhs);
    ops_.push_back(rhs);
    set_loc(lhs->pos1(), rhs->pos2());
}

PostfixExpr::PostfixExpr(const Expr* lhs, Kind kind, const Position& pos2) 
    : kind_(kind)
{
    ops_.push_back(lhs);
    set_loc(lhs->pos1(), pos2);
}

IndexExpr::IndexExpr(const Position& pos1, const Expr* lhs, const Expr* index, const Position& pos2) {
    ops_.push_back(lhs);
    ops_.push_back(index);
    set_loc(pos1, pos2);
}

Call::Call(const Expr* fct) { ops_.push_back(fct); }
bool Call::is_continuation_call() const { return type()->isa<NoRet>(); }

void Call::set_pos2(const Position& pos2) {
    assert(!ops_.empty());
    HasLocation::set_loc(ops_.front()->pos1(), pos2);
}

Location Call::args_location() const {
    if (ops().size() == 1)
        return Location(pos2());
    return Location(op(1)->pos1(), ops_.back()->pos2());
}

/*
 * Stmt
 */

ExprStmt::ExprStmt(const Expr* expr, const Position& pos2)
    : expr_(expr)
{
    set_loc(expr->pos1(), pos2);
}

DeclStmt::DeclStmt(const VarDecl* var_decl, const Expr* init, const Position& pos2)
    : var_decl_(var_decl)
    , init_(init)
{
    set_loc(var_decl->pos1(), pos2);
}

IfElseStmt::IfElseStmt(const Position& pos1, const Expr* cond, const Stmt* thenStmt, const Stmt* elseStmt)
    : cond_(cond)
    , thenStmt_(thenStmt)
    , elseStmt_(elseStmt)
{
    set_loc(pos1, elseStmt->pos2());
}

void DoWhileStmt::set(const Position& pos1, const Stmt* body, const Expr* cond, const Position& pos2) {
    Loop::set(cond, body);
    set_loc(pos1, pos2);
}

void ForStmt::set(const Position& pos1, const Expr* cond, const Expr* step, const Stmt* body) {
    Loop::set(cond, body);
    step_ = step;
    set_loc(pos1, body->pos2());
}

bool ForStmt::is_while() const { 
    if (const ExprStmt* expr_stmt = init()->isa<ExprStmt>())
        return expr_stmt->expr()->isa<EmptyExpr>() && step()->isa<EmptyExpr>();
    return false;
}

void ForStmt::set_empty_init(const Position& pos) {
    set(new ExprStmt(new EmptyExpr(pos), pos));
}

BreakStmt::BreakStmt(const Position& pos1, const Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    set_loc(pos1, pos2);
}

ContinueStmt::ContinueStmt(const Position& pos1, const Position& pos2, const Loop* loop) 
    : loop_(loop)
{
    set_loc(pos1, pos2);
}

ReturnStmt::ReturnStmt(const Position& pos1, const Expr* expr, const Fct* fct, const Position& pos2)
    : expr_(expr)
    , fct_(fct)
{
    set_loc(pos1, pos2);
}

} // namespace impala
