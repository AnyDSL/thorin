#include "impala/ast.h"

#include "anydsl/cfg.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/var.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

#include "impala/type.h"

using anydsl::BB;
using anydsl::Def;
using anydsl::Var;
using anydsl::LVar;
using anydsl::RVar;
using anydsl::make_name;

namespace impala {

class CodeGen {
public:

    BB* curBB;
    anydsl::Fct* curFct;

    anydsl::World& world;
};

//------------------------------------------------------------------------------

void Prg::emit(CodeGen& cg) const {
    for_all (f, fcts())
        f->emit(cg);
}

void Fct::emit(CodeGen& cg) const {
}

void Decl::emit(CodeGen& cg) const {
}

/*
 * Expr
 */

Var EmptyExpr::emit(CodeGen& cg) const {
    return 0;
}

Var Literal::emit(CodeGen& cg) const {
    return RVar(cg.world.literal((anydsl::PrimLitKind) kind(), value()));
}

Var Id::emit(CodeGen& cg) const {
    return cg.curBB->getVar(symbol());
}

Var PrefixExpr::emit(CodeGen& cg) const {
    Var bvar = rexpr()->emit(cg);
    const anydsl::PrimType* p = bvar.type()->as<anydsl::PrimType>();

    if (kind() == ADD)
        return bvar; // this is a NOP

    switch (kind()) {
        case SUB: {
            // TODO incorrect for f32, f64
            const anydsl::PrimLit* zero = cg.world.literal(p->kind(), 0u);
            return RVar(cg.world.createArithOp(anydsl::ArithOp_sub, zero, bvar.load()));
        }
        case INC:
        case DEC: {
            const anydsl::PrimLit* one = cg.world.literal(p->kind(), 1u);
            RVar val = RVar(cg.world.createArithOp(Token::toArithOp((TokenKind) kind()), bvar.load(), one));
            bvar.store(val.load());
            return bvar;
        }
        default: ANYDSL_UNREACHABLE; // TODO
    }
}

static Var emitInfix(TokenKind op, Var avar, Var bvar, anydsl::World& world) {
    if (Token::isAsgn(op)) {
        Var var = bvar;

        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            var = emitInfix(sop, avar, bvar, world);
        }

        avar.store(var.load());
        return avar;
    }

    if (Token::isArith(op))
        return RVar(world.createArithOp(Token::toArithOp(op), avar.load(), bvar.load()));

    anydsl_assert(Token::isRel(op), "must be a relop");
    return RVar(world.createRelOp(Token::toRelOp(op), avar.load(), bvar.load()));
}

Var InfixExpr::emit(CodeGen& cg) const {
    Var avar = lexpr()->emit(cg);
    Var bvar = rexpr()->emit(cg);

    return emitInfix((TokenKind) kind(), avar, bvar, cg.world);
}

Var PostfixExpr::emit(CodeGen& cg) const {
    return 0;
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    decl()->emit(cg);
}

void ExprStmt::emit(CodeGen& cg) const {
    expr()->emit(cg);
}

void IfElseStmt::emit(CodeGen& cg) const {
    static int id = 0;

    // always create elseBB -- the edge from headBB to nextBB would be crtical anyway

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("if-head", id));
    BB* thenBB = cg.curFct->createBB(make_name("if-then", id));
    BB* elseBB = cg.curFct->createBB(make_name("if-else", id));
    BB* nextBB = cg.curFct->createBB(make_name("if-next", id));

    // head
    cg.curBB->fixto(headBB);
    cg.curBB = headBB;

    // condition
    Var cvar = cond()->emit(cg);
    headBB->branches(cvar.load(), thenBB, elseBB);
    headBB->seal();
    thenBB->seal();
    elseBB->seal();

    // then
    cg.curBB = thenBB;
    thenStmt()->emit(cg);
    cg.curBB->fixto(nextBB);

    // else
    cg.curBB = elseBB;
    elseStmt()->emit(cg);
    cg.curBB->fixto(nextBB);

    // next
    cg.curBB = nextBB;
    nextBB->seal();

    ++id;
}

void WhileStmt::emit(CodeGen& cg) const {
    static int id = 0;

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("while-head", id));
    BB* bodyBB = cg.curFct->createBB(make_name("while-body", id));
    BB* nextBB = cg.curFct->createBB(make_name("while-next", id));

    // head
    cg.curBB->fixto(headBB);
    cg.curBB = headBB;

    // condition
    Var cvar = cond()->emit(cg);
    headBB->branches(cvar.load(), bodyBB, nextBB);
    bodyBB->seal();

    // body
    cg.curBB = bodyBB;
    body()->emit(cg);
    cg.curBB->fixto(headBB);
    headBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();

    ++id;
}

void DoWhileStmt::emit(CodeGen& cg) const {
    body()->emit(cg);
    cond()->emit(cg);
}

void ForStmt::emit(CodeGen& cg) const {
    init()->emit(cg);
    cond()->emit(cg);
    inc()->emit(cg);
    body()->emit(cg);
}

void BreakStmt::emit(CodeGen& cg) const {
}

void ContinueStmt::emit(CodeGen& cg) const {
}

void ReturnStmt::emit(CodeGen& cg) const {
    expr()->emit(cg);
}

void ScopeStmt::emit(CodeGen& cg) const {
    for_all (const &s, stmts())
        s->emit(cg);
}

//------------------------------------------------------------------------------

/*
 * Type
 */

const anydsl::Type* PrimType::emit(anydsl::World& world) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) \
        case TYPE_##itype: \
            return world.type_##atype();
#include "impala/tokenlist.h"
        default:
            ANYDSL_UNREACHABLE;
    }
}

const anydsl::Type* Void::emit(anydsl::World& world) const {
    return world.unit();
}

const anydsl::Type* TypeError::emit(anydsl::World& /*world*/) const {
    ANYDSL_UNREACHABLE;
}

} // namespace impala
