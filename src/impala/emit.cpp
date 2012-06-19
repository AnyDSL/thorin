#include "impala/ast.h"

#include "anydsl/cfg.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/var.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

#include "impala/type.h"

using anydsl::Var;
using anydsl::LVar;
using anydsl::RVar;

namespace impala {

class CodeGen {
public:

    anydsl::BB* bb;

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
    return *cg.bb->getVN(symbol());
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

#if 0
anydsl::PrimTypeKind Token::toPrimType() const {
    switch (kind_) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_ ## itype: return anydsl::PrimType_##atype;
#include "impala/tokenlist.h"
        default: ANYDSL_UNREACHABLE;
    }
}
#endif


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



#if 0
void Parser::parseFct() {
#if 0
    emit.pushScope();

    eat(Token::DEF);
    Token fname = parseId();
    fcts_[fname.symbol()] = emit.fct(fname);

    expect(Token::L_PAREN, "function head");
    PARSE_COMMA_LIST
    (
        parseParam(),
        Token::R_PAREN,
        "arguments of a function call"
    )

    // return-continuation
    if (accept(Token::ARROW))
        curFct()->setReturnCont(parseType());

    parseScopeBody();
    emit.fixto(curFct()->exit());

#if 0
    fct->finalizeAll();

    // HACK
    Param* retVal = new Param(retType->loc());
    retVal->meta = retType;
    fct->exit_->lambda_->params().push_back(retVal);
    fct->exit_->beta_->args().push_back(retVal);
#endif

    emit.popScope();
#endif
}

void Parser::parseIfElse() {
    eat(Token::IF);
    emit.pushScope();
    int id = nextId();

    Location loc(la().loc());

    BB*  oldBB = curBB();
    BB*   ifBB = curFct()->createBB(make_name("if-then", id));
    BB* elseBB = curFct()->createBB(make_name("if-else", id));
    BB* contBB = curFct()->createBB(make_name("if-cont", id));

    Def* cond = parseCond("if-statement").load();

    oldBB->branches(cond, ifBB, elseBB);

    emit.curBB = ifBB;
    parseScope();

    if (accept(Token::ELSE)) {
        emit.popScope();
        emit.fixto(contBB);
        emit.curBB = elseBB;
        emit.pushScope();
        parseScope();
    }

    emit.fixto(contBB); 
    emit.popScope();
}
const Stmt* Parser::parseWhile() {
    eat(Token::WHILE);
    emit.pushScope();
    int id = nextId();

    BB* headBB = curFct()->createBB(make_name("while-head", id));
    BB* bodyBB = curFct()->createBB(make_name("while-body", id));
    BB* iterBB = curFct()->createBB(make_name("while-iter", id));
    BB* contBB = curFct()->createBB(make_name("while-cont", id));

    emit.fixto(headBB);
    Def* cond = parseCond("while-statement").load();

    headBB->branches(cond, bodyBB, contBB);
    iterBB->goesto(headBB);

    emit.curBB = bodyBB;
    parseScope();
    emit.fixto(iterBB);

    emit.popScope();
    emit.curBB = contBB;
}

const Stmt* Parser::parseDoWhile() {
    eat(Token::DO);
    emit.pushScope();
    int id = nextId();

    BB* bodyBB = curFct()->createBB(make_name("do-body", id));
    BB* iterBB = curFct()->createBB(make_name("do-iter", id));
    BB* contBB = curFct()->createBB(make_name("do-cont", id));

    emit.fixto(bodyBB);
    iterBB->goesto(bodyBB);

    parseScope();
    emit.fixto(iterBB);
    expect(Token::WHILE, "do-while-statement");
    parseCond("do-while-statement");
    expect(Token::SEMICOLON, "do-while-statement");

    emit.popScope();
    emit.curBB = contBB;
}

#if 0
const Stmt* Parser::parseFor() {
    int id = nextId();
    emit.pushScope();
    Location loc;

    BB* prehBB = curFct()->createBB(make_name("for-preh", id));
    BB* headBB = curFct()->createBB(make_name("for-head", id));
    BB* bodyBB = curFct()->createBB(make_name("for-body", id));
    BB* iterBB = curFct()->createBB(make_name("for-iter", id));
    BB* contBB = curFct()->createBB(make_name("for-cont", id));

    headBB->setMulti();

    eat(Token::FOR);
    expect(Token::L_PAREN, "for-statement");

    emit.fixto(prehBB);

    // clause 1: decl or expr_opt ';'
    if (la2() == Token::COLON)
        parseDeclStmt();
    else if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
    } else if (isExpr()) {
        parseExprStmt();
    } else
        error("expression or delcaration-statement", 
                "first clause in for-statement");

    emit.fixto(headBB);
        
    // clause 2: expr_opt ';'
    Def* cond;
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
        // but create true cond
        cond = world().literal(true);
    } else if (isExpr()) {
        cond = parseExpr().load();
        expect(Token::SEMICOLON, "second clause in for-statement");
    } else {
        error("expression or nothing", 
                "second clause in for-statement");
        cond = world().literal_error(world().type_u1());
    }

    emit.curBB = iterBB;

    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
    } else if (isExpr()) {
        parseExpr();
        expect(Token::R_PAREN, "for-statement");
    } else {
        error("expression or nothing",
                "third clause in for-statement");
    }

    emit.curBB = bodyBB;
    parseScope();
    emit.fixto(iterBB);

    headBB->branches(cond, bodyBB, contBB);
    iterBB->goesto(headBB);

    emit.curBB = contBB;
    emit.popScope();
}

const Stmt* Parser::parseReturn() {
    Token ret = eat(Token::RETURN);

    Value res;
    if (accept(Token::SEMICOLON))
        res = Value(); // TODO
    else if (isExpr()) {
        res = parseExpr();
        expect(Token::SEMICOLON, "return-statement");
    }

    emit.returnStmt(res);
}
#endif
#endif

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
    cond()->emit(cg);
    ifStmt()->emit(cg);
    elseStmt()->emit(cg);
}

void WhileStmt::emit(CodeGen& cg) const {
    cond()->emit(cg);
    body()->emit(cg);
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

const anydsl::Type* PrimType::toAnyDSL(anydsl::World& world) const {
#if 0
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) case TYPE_##itype: p.o << #itype; return;
#include "impala/tokenlist.h"
    }
#endif
    return 0;
}

const anydsl::Type* Void::toAnyDSL(anydsl::World& world) const {
    return 0;
}

const anydsl::Type* TypeError::toAnyDSL(anydsl::World& world) const {
    return 0;
}

} // namespace impala
