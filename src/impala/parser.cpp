#include "impala/parser.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "anydsl/util/assert.h"
#include "anydsl/support/cfg.h"
#include "anydsl/support/symbol.h"

#include "impala/binding.h"

#if 0

#define PARSE_COMMA_LIST(stmnt, delimiter, context) { \
        if (la() != delimiter) { \
            do { stmnt; } \
            while ( accept(Token::COMMA) ); \
        } \
        expect(delimiter, context); \
    }

using namespace anydsl;

namespace impala {

/*
 * statics
 */

/*static*/ Type2Prec    Parser::preRPrec_;
/*static*/ Type2BinPrec Parser::binPrec_;
/*static*/ Type2Prec    Parser::postLPrec_;

/*
 * constructor and destructor
 */

Parser::Parser(std::istream& stream, const std::string& filename)
    : lexer_(stream, filename)
    , break_(0)
    , continue_(0)
    , counter_(0)
{
    Token::init(); // HACK

    // init 2 lookahead
    lookahead_[0] = lexer_.lex();
    lookahead_[1] = lexer_.lex();

    static bool init = false;
    init = true;

    if (!init) {
        // init prec tables
#define IMPALA_PREFIX(    tok, t_str,    r)  preRPrec_[Token:: tok] = r;
#define IMPALA_POSTFIX(   tok, t_str, l   ) postLPrec_[Token:: tok] = l;
#define IMPALA_INFIX(     tok, t_str, l, r)   binPrec_[Token:: tok] = BinPrec(l, r);
#define IMPALA_INFIX_ASGN(tok, t_str, l, r)   binPrec_[Token:: tok] = BinPrec(l, r);
#include <impala/tokenlist.h>
    }
}

/*
 * helpers
 */

Token Parser::lex() {
    Token result  = lookahead_[0]; // remember result
    lookahead_[0] = lookahead_[1]; // copy over LA2 to LA1
    lookahead_[1] = lexer_.lex();  // fill new LA2

    return result;
}

int Parser::nextId() {
    int id = counter_++;
    counter_ %= 0x100;

    return id;
}

static Symbol make_symbol(const char* cstr, int id) {
    std::ostringstream oss;
    oss << '<' << cstr << '-';
    oss << std::setw(2) << std::setfill('0') << id << '>';

    return Symbol(oss.str());
}

bool Parser::accept(TokenType type) {
    if (type == la()) {
        lex();
        return true;
    } else
        return false;
}

bool Parser::expect(TokenType tok, const std::string& context) {
    if (la() == tok) {
        lex();
        return true;
    } else {
        std::ostringstream oss;
        oss << '\'' << tok << '\'';
        error(oss.str(), context);
        return false;
    }
}

void Parser::error(const std::string& what, const std::string& context) {
    std::ostream& os = la().error() << "expected " << what << ", got '" << la() << "'";

    if (!context.empty())
        os << " while parsing " << context;

    os << "\n";
}

/*
 * misc
 */

Lambda* Parser::parse() {
    emit.prologue(la().pos1());
    parseGlobals();

    return emit.exit();
}

Token Parser::parseId() {
    Token name;
    if (la() == Token::ID)
        name = lex();
    else
        name = Token(la().loc(), make_symbol("error-name", nextId()).str());

    return name;
}

Type* Parser::parseType() {
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_ ## itype: \
            return emit.builtinType(lex());
#include <impala/tokenlist.h>
            
        default: ANYDSL_UNREACHABLE; // TODO
    }
}

Type* Parser::parseParam() {
    Token name = parseId();
    expect(Token::COLON, "lambda parameter");
    Type* type = parseType();
    Param* param = new Param(Location(name.loc().pos1(), type->pos2()), name.symbol());
    param->meta = type;
    emit.param(name, type, param);

    return type;
}

Value Parser::parseDecl() {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
    //Value type = tryExpr();
    Type* type = parseType();

    return emit.decl(tok, type);
}

void Parser::parseGlobals() {
    while (true) {
        switch (la()) {
            case Token::DEF:         parseFct(); continue;
            case Token::END_OF_FILE: return;

            // consume token nobody wants to have in order to prevent infinite loop
            default:                  lex(); 
                error("global", "global list");
                lex(); 
        }
    }
}

void Parser::parseFct() {
    emit.pushScope();
    Cursor old;

    Location loc(la().loc());
    eat(Token::DEF);
    Token fname = parseId();
    Fct* fct = emit.fct(old, la().loc(), fname);
    //ParamList& params = fct->lambda_->params();
    expect(Token::L_PAREN, "function head");
    PARSE_COMMA_LIST
    (
        {
            Type* type = parseParam();
            fct->lambda_->appendParam(type, type->loc());
        },
        //params.push_back(scast<Param>(parseParam().load())),
        Token::R_PAREN,
        "arguments of a function call"
    )
    Type* retType = 0;

    // return-continuation
    if (accept(Token::ARROW)) {
        retType = parseType();
        fct->setReturn(Location(/*todo*/), retType);
    }

    parseScopeBody();

    if (!emit.getCursor().bb->lcursor_)
        anydsl_assert(true, "todo");

    emit.getCursor().bb->hangInBB(fct->exit_);
    emit.glueTo(loc, fct->exit_);

    fct->finalizeAll();

    // HACK
    Param* retVal = new Param(retType->loc());
    retVal->meta = retType;
    fct->exit_->lambda_->params().push_back(retVal);
    fct->exit_->beta_->args().push_back(retVal);

    emit.setCursor(old);
    emit.popScope();
}

void Parser::parseStmtList() {
    while (true) {
        if (isExpr()) {
            parseStmt();
            continue;
        }

        switch (la()) {
#define IMPALA_KEY_STMT(tok, t_str) case Token:: tok:
#include <impala/tokenlist.h>
            case Token::L_PAREN:
            case Token::L_BRACE:  
            case Token::ID:          parseStmt(); continue;

            case Token::END_OF_FILE: return;
            case Token::R_BRACE:     return;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("statement", "statement list");
                lex(); 
        }
    }
}

/*
 * statements
 */

void Parser::parseStmt() {
    if (isExpr())
        return parseExprStmt();

    if (accept(Token::ELSE)) {
        error("'else' without matching 'if'", "statement");
        return;
    }

    switch (la()) {
        // expression or a decl
        case Token::ID:
            if (la2() == Token::COLON)
                return parseDeclStmt();
            else {
                error("statement", "");
                return;
            }

        // other statements
        case Token::IF:        return parseIfElse();
        case Token::WHILE:     return parseWhile();
        case Token::DO:        return parseDoWhile();
        case Token::FOR:       return parseFor();
        case Token::BREAK:     return parseBreak();
        case Token::CONTINUE:  return parseContinue();
        case Token::RETURN:    return parseReturn();
        case Token::L_BRACE:   return parseCompoundStmt();
        case Token::SEMICOLON: return; // empty statement
        default:               error("statement", "");
    }
}

void Parser::parseExprStmt() {
    parseExpr(); // discard val
    expect(Token::SEMICOLON, "the end of an expression statement");
}

void Parser::parseDeclStmt() {
    Value aval = parseDecl();

    // initialization
    if (la() == Token::COLONEQ) {
        Token op = Token(lex().loc(), Token::ASGN);
        Value bval = tryExpr();
        emit.infixOp(aval, op, bval);
    }

    expect(Token::SEMICOLON, "the end of an declaration statement");
}

void Parser::parseIfElse() {
    int id = nextId();

    emit.pushScope();

    Location loc(la().loc());

    BB*  oldBB = emit.getCursor().bb;
    BB*   ifBB = oldBB->createSubBB(loc, make_symbol("if-body", id));
    BB* elseBB = oldBB->createSubBB(loc, make_symbol("if-else-next", id));
    BB* nextBB = 0; // only if else is there, other wise elseBB is infact "next"

    eat(Token::IF);
    Def* cond = parseCond("if-statement").load();

    oldBB->branches(loc, cond, ifBB, elseBB);

    emit.setCursor(ifBB);
    parseScope();

    if (accept(Token::ELSE)) {
        emit.popScope();
        nextBB = oldBB->createSubBB(loc, make_symbol("next-body", id));
        emit.fixBB(loc, nextBB);
        emit.setCursor(elseBB);
        emit.pushScope();
        parseScope();
        emit.glueTo(loc, nextBB);
    } else {
        emit.glueTo(loc, elseBB); 
        elseBB->param_->symbol() = make_symbol("next-body", id);
        emit.setCursor(elseBB);
    }

    emit.popScope();
}

void Parser::parseWhile() {
    int id = nextId();
    Location loc(la().loc());

    emit.pushScope();

    BB*  oldBB = emit.getCursor().bb;
    BB* headBB =  oldBB->createSubBB(loc, make_symbol("while-head", id));
    BB* bodyBB = headBB->createSubBB(loc, make_symbol("while-body", id));
    BB* nextBB = headBB->createSubBB(loc, make_symbol("while-next", id));

    eat(Token::WHILE);
    Def* cond = parseCond("while-statement").load();

    emit.setCursor(bodyBB);
    parseScope();
    emit.glueTo(loc, nextBB);

    oldBB->jumps(loc, headBB);
    headBB->branches(loc, cond, bodyBB, nextBB);
    bodyBB->jumps(loc, headBB);

    emit.popScope();
}

void Parser::parseDoWhile() {
    emit.pushScope();

    eat(Token::DO);
    parseScope();
    expect(Token::WHILE, "do-while-statement");
    parseCond("do-while-statement");
    expect(Token::SEMICOLON, "do-while-statement");

    emit.popScope();
}

void Parser::parseFor() {
    int id = nextId();
    emit.pushScope();
    Location loc;

    BB* old_BB = emit.getCursor().bb;
    BB* pre_BB = old_BB->createSubBB(loc, make_symbol("for-pre", id));
    BB* headBB = pre_BB->createSubBB(loc, make_symbol("for-head", id));
    BB* bodyBB = pre_BB->createSubBB(loc, make_symbol("for-body", id));
    BB* iterBB = bodyBB->createSubBB(loc, make_symbol("for-iter", id));
    BB* nextBB = pre_BB->createSubBB(loc, make_symbol("for-next", id));

    old_BB->flowsTo(pre_BB);
    pre_BB->flowsTo(headBB);
    headBB->flowsTo(bodyBB);
    headBB->flowsTo(nextBB);
    bodyBB->flowsTo(iterBB);
    iterBB->flowsTo(headBB);

    eat(Token::FOR);
    expect(Token::L_PAREN, "for-statement");

    emit.setCursor(pre_BB);

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

    emit.glueTo(loc, headBB);
        
    // clause 2: expr_opt ';'
    Def* cond;
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
        // but create true cond
        cond = Primitive::createBoolean(true);
    } else if (isExpr()) {
        cond = parseExpr().load();
        expect(Token::SEMICOLON, "second clause in for-statement");
    } else {
        error("expression or nothing", 
                "second clause in for-statement");
        cond = new ErrorValue(la().loc());
    }

    emit.setCursor(iterBB);

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

    emit.setCursor(bodyBB);
    parseScope();
    emit.glueTo(loc, iterBB);

    old_BB->jumps(loc, pre_BB);
    headBB->branches(loc, cond, bodyBB, nextBB);
    iterBB->jumps(loc, headBB);

    emit.setCursor(nextBB);
    emit.popScope();
}

void Parser::parseBreak() {
    eat(Token::BREAK);
    expect(Token::SEMICOLON, "break-statement");
}

void Parser::parseContinue() {
    eat(Token::CONTINUE);
    expect(Token::SEMICOLON, "continue-statement");
}

void Parser::parseReturn() {
    Token ret = eat(Token::RETURN);

    Value res;
    if (accept(Token::SEMICOLON))
        res = Value(); // TODO
    else if (isExpr()) {
        res = parseExpr();
        expect(Token::SEMICOLON, "return-statement");
    }

    emit.returnStmt(Location(ret.pos1(), res.pos2()), res);
}

void Parser::parseScopeBody() {
    eat(Token::L_BRACE);
    parseStmtList();
    expect(Token::R_BRACE, "scope-statement");
}

void Parser::parseScope() {
    if (la() == Token::L_BRACE)
        parseScopeBody();
    else
        parseStmt();
}

void Parser::parseCompoundStmt() {
    emit.pushScope();
    parseScopeBody();
    emit.popScope();
}

Value Parser::parseCond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    Value val = tryExpr();
    expect(Token::R_PAREN, "condition in " + what);

    return val;
}

bool Parser::isExpr() {
    // identifier without a succeeding colon
    if (la() == Token::ID && la2() != Token::COLON)
        return true;

    switch (la()) {
#define IMPALA_PREFIX(tok, t_str, r) case Token:: tok:
#define IMPALA_KEY_EXPR(tok, t_str)  case Token:: tok:
#define IMPALA_LIT(tok, t)           case Token:: tok:
//#define IMPALA_TYPE(itype, atype)    case Token:: TYPE_ ## itype:
#include <impala/tokenlist.h>
        case Token::L_PAREN:
            return true;
        default:
            return false;
    }
}

Value Parser::tryExpr() {
    return isExpr() ? parseExpr() : Value(new ErrorValue(la().loc()));
}

/*
 * expressions
 */

Value Parser::parseExpr(Prec prec) {
    Value aval;

    if (la().isPrefix())
        aval = parsePrefixExpr();
    else
        aval = parsePrimaryExpr();

    while (true) {
        /*
         * (aval  op  LA) op ...  on break  (current prec > aval prec of LA)  -->  reduce
         *  aval  op (LA  op ...) otherwise                                   -->  shift
         */

        if (la().isInfix()) {
            if ( prec > binPrec_[la()].l )
                break;

            aval = parseInfixExpr(aval);
        } else if ( la().isPostfix() ) {
            if ( prec > postLPrec_[la()] )
                break;

            aval = parsePostfixExpr(aval);
        } else
            break;
    }

    return aval;
}

Value Parser::parsePrefixExpr() {
    Token op = lex();
    Value bval = parseExpr(preRPrec_[op]);

    return emit.prefixOp(op, bval);
}

Value Parser::parseInfixExpr(Value aval) {
    Token op = lex();
    Value bval = parseExpr(binPrec_[op].r);

    return emit.infixOp(aval, op, bval);
}

Value Parser::parsePostfixExpr(Value aval) {
    if (accept(Token::L_PAREN)) {
        std::vector<Value> args;
        PARSE_COMMA_LIST
        (
            args.push_back(tryExpr()),
            Token::R_PAREN,
            "arguments of a function call"
        )

        return emit.fctCall(aval, args);
    } else
        return emit.postfixOp(aval, lex());
}

Value Parser::parsePrimaryExpr() {
    switch (la() ) {
        case Token::L_PAREN: {
            eat(Token::L_PAREN);
            Value val = parseExpr();
            expect(Token::R_PAREN, "primary expression");
            return val;
        }
        case Token::ID:     return emit.id(lex());
        case Token::LAMBDA: return parseLambda();
        //case Token::PI:            return parsePi();
        //case Token::SIGMA:         return parseSigma();
        //case Token::MEMORY:        return parseMemory();

#define IMPALA_LIT(tok, t) case Token:: tok:
#include <impala/tokenlist.h>
        case Token::TRUE:
        case Token::FALSE:
            return parseLiteral();

//#define IMPALA_TYPE(itype, atype) case Token:: TYPE_ ## itype:
//#include <impala/tokenlist.h>
            //return Value(parseType());

        default: {
            error("expression", "primary expression");
            return Value(new ErrorValue(la().loc()));
        }
    }
}

Value Parser::parseLiteral() {
    return emit.literal(lex());
}

Value Parser::parseLambda() {
    emit.pushScope();

    eat(Token::LAMBDA);
    expect(Token::L_PAREN, "lambda");
    PARSE_COMMA_LIST
    (
        parseDecl(),
        Token::R_PAREN,
        "lambda parameters"
    )
    expect(Token::ARROW, "lambda");
    parseScopeBody();

    emit.popScope();

    return Value() /*TODO*/;
}

} // namespace impala
#endif
