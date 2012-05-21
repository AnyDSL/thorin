#include "impala/parser.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "anydsl/air/literal.h"
#include "anydsl/air/world.h"
#include "anydsl/support/binding.h"
#include "anydsl/support/cfg.h"
#include "anydsl/support/symbol.h"
#include "anydsl/util/assert.h"


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

Parser::Parser(anydsl::World& world, std::istream& stream, const std::string& filename)
    : lexer_(stream, filename)
    , emit(world)
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

Parser::~Parser() {
    FOREACH(p, fcts_) delete p.second;
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

static std::string make_name(const char* cstr, int id) {
    std::ostringstream oss;
    oss << '<' << cstr << '-';
    oss << std::setw(2) << std::setfill('0') << id << '>';

    return oss.str();
}

static Symbol make_symbol(const char* cstr, int id) {
    return Symbol(make_name(cstr, id));
}

bool Parser::accept(TokenKind type) {
    if (type == la()) {
        lex();
        return true;
    } else
        return false;
}

bool Parser::expect(TokenKind tok, const std::string& context) {
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
#include "impala/tokenlist.h"
            
        default: ANYDSL_UNREACHABLE; // TODO
    }
}

void Parser::parseParam() {
    Token name = parseId();
    expect(Token::COLON, "lambda parameter");
    const Type* type = parseType();

    emit.param(name, type);
}

Value Parser::parseDecl() {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
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
        //case Token::FOR:       return parseFor();
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

void Parser::parseWhile() {
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

void Parser::parseDoWhile() {
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
void Parser::parseFor() {
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
#endif

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

    emit.returnStmt(res);
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
    return isExpr() ? parseExpr() : emit.error();
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
            return emit.error();
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
