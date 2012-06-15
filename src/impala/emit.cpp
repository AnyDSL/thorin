#include "impala/ast.h"

#include "anydsl/type.h"
#include "anydsl/util/for_all.h"

namespace impala {

class CodeGen {
public:

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

void EmptyExpr::emit(CodeGen& cg) const {
}

void Literal::emit(CodeGen& cg) const {
}

void Id::emit(CodeGen& cg) const {
}

void PrefixExpr::emit(CodeGen& cg) const {
}

void InfixExpr::emit(CodeGen& cg) const {
    lexpr()->emit(cg);
    rexpr()->emit(cg);

#if 0
    const PrimType* p1 = lval.type()->isa<PrimType>();
    const PrimType* p2 = rval.type()->isa<PrimType>();

    if (p1 && p1 == p2) {
        if (op.isAsgn()) {
            Value val = bval;

            if (op != Token::ASGN) {
                Token tok = op.seperateAssign();
                val = infixOp(aval, tok, bval);
            }

            aval.store(val.load());
            return aval;
        }

        if (op.isArith())
            return Value(world.createArithOp(op.toArithOp(), aval.load(), bval.load()));
        else if (op.isRel())
            return Value(world.createRelOp(op.toRelOp(), aval.load(), bval.load()));
    }

    op.error() << "type error: TODO\n";

    const Type* t = p1 ? p1 : p2;
    if (t)
        return Value(world.literal_error(t)); 

    return error();
#endif
}

void PostfixExpr::emit(CodeGen& cg) const {
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


} // namespace impala
