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
