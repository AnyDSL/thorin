#include "impala/parser.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "anydsl/util/assert.h"

#include "impala/ast.h"
#include "impala/lexer.h"
#include "impala/prec.h"
#include "impala/type.h"

using namespace anydsl;

namespace impala {

class Parser {
public:

    /*
     * constructor
     */

    Parser(TypeTable&, std::istream& stream, const std::string& filename);

    /*
     * helpers
     */

    const Token& la () const { return lookahead_[0]; }
    const Token& la2() const { return lookahead_[1]; }

#ifdef NDEBUG
    Token eat(TokenKind /*what*/) { return lex(); }
#else
    Token eat(TokenKind what) { assert(what == la() && "internal parser error"); return lex(); }
#endif

    bool accept(TokenKind tok);

    bool expect(TokenKind tok, const std::string& context);
    void error(const std::string& what, const std::string& context);
    bool result() const { return result_; }

    // misc
    const Prg* parse();
    Token parseId();
    const Type* parseType();
    const Decl* parseDecl();
    void parseGlobals();
    const Fct* parseFct();

    // expressions
    bool isExpr();
    const Expr* tryExpr();
    const Expr* parseExpr(Prec prec);
    const Expr* parseExpr() { return parseExpr(BOTTOM); }
    const Expr* parsePrefixExpr();
    const Expr* parseInfixExpr(const Expr* lhs);
    const Expr* parsePostfixExpr(const Expr* lhs);
    const Expr* parsePrimaryExpr();
    const Expr* parseLiteral();
    const Expr* parseLambda();

    // statements
    const Stmt* parseStmt();
    const ExprStmt* parseExprStmt();
    const DeclStmt* parseDeclStmt();
    const Stmt* parseIfElse();
    const Stmt* parseWhile();
    const Stmt* parseDoWhile();
    const Stmt* parseFor();
    const Stmt* parseBreak();
    const Stmt* parseContinue();
    const Stmt* parseReturn();

    /// { Stmt* }
    const ScopeStmt* parseScope();

    /// helper for condition in if/while/do-while
    const Expr* parseCond(const std::string& what);

private:

    int nextId();

    /// Consume next Token in input stream, fill look-ahead buffer, return consumed Token.
    Token lex();

    /*
     * data
     */

    TypeTable& types;
    Lexer lexer_;       ///< invoked in order to get next token
    Token lookahead_[2];///< LL(2) look ahead
    const Loop* curLoop_;
    const Fct* curFct_;
    Prg* prg_;
    int counter_;
    anydsl::Location prevLoc_;
    bool result_;
};

//------------------------------------------------------------------------------

const Prg* parse(TypeTable& types, std::istream& i, const std::string& filename, bool& result) {
    Parser p(types, i, filename);
    const Prg* prg = p.parse();
    result = p.result();

    return prg;
}

//------------------------------------------------------------------------------

#define PARSE_COMMA_LIST(stmnt, delimiter, context) { \
        if (la() != delimiter) { \
            do { stmnt; } \
            while ( accept(Token::COMMA) ); \
        } \
        expect(delimiter, context); \
    }


/*
 * constructor and destructor
 */

Parser::Parser(TypeTable& types, std::istream& stream, const std::string& filename)
    : types(types)
    , lexer_(stream, filename)
    , curLoop_(0)
    , curFct_(0)
    , prg_(new Prg())
    , counter_(0)
    , result_(true)
{
    // init 2 lookahead
    lookahead_[0] = lexer_.lex();
    lookahead_[1] = lexer_.lex();
}

/*
 * helpers
 */

Token Parser::lex() {
    Token result  = lookahead_[0]; // remember result
    lookahead_[0] = lookahead_[1]; // copy over LA2 to LA1
    lookahead_[1] = lexer_.lex();  // fill new LA2
    prevLoc_ = result.loc();       // remember previous location

    return result;
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
    result_ = false;

    std::ostream& os = la().emitError() << "expected " << what << ", got '" << la() << "'";

    if (!context.empty())
        os << " while parsing " << context;

    os << "\n";
}

int Parser::nextId() {
    int id = counter_++;
    counter_ %= 0x100;

    return id;
}

/*
 * misc
 */

const Prg* Parser::parse() {
    parseGlobals();
    return prg_;
}

Token Parser::parseId() {
    Token name;
    if (la() == Token::ID)
        name = lex();
    else
        name = Token(la().loc(), make_symbol("error-name", nextId()).str());

    return name;
}

const Type* Parser::parseType() {
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token::TYPE_ ## itype: lex(); return types.type_##itype();
#include "impala/tokenlist.h"

        case Token::TYPE_int:  lex(); return types.type_int32();
        case Token::TYPE_uint: lex(); return types.type_uint32();
        case Token::TYPE_void: lex(); return types.type_void();
            
        default: ANYDSL_UNREACHABLE; // TODO
    }
}

const Decl* Parser::parseDecl() {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
    const Type* type = parseType();

    return new Decl(tok, type, prevLoc_.pos2());
}

void Parser::parseGlobals() {
    while (true) {
        switch (la()) {
            case Token::DEF:         prg_->fcts_.push_back(parseFct()); continue;
            case Token::END_OF_FILE: return;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("global", "global list");
                lex(); 
        }
    }
}

const Fct* Parser::parseFct() {
    Position pos1 = eat(Token::DEF).pos1();
    Token id = parseId();

    Fct* fct = new Fct();
    curFct_ = fct;
    const Type* retType = 0;

    std::vector<const Type*> argTypes;

    expect(Token::L_PAREN, "function head");
    PARSE_COMMA_LIST
    (
        {
            const Decl* param = parseDecl();
            fct->params_.push_back(param);
            argTypes.push_back(param->type());
        },
        Token::R_PAREN,
        "arguments of a function call"
    )

    // return-continuation
    if (accept(Token::ARROW))
        retType = parseType();
    else
        retType = types.noret();

    const Pi* pi = types.pi(argTypes.begin().base(), argTypes.end().base(), retType);
    const Decl* decl = new Decl(id, pi, prevLoc_.pos2());
    const ScopeStmt* body = parseScope();

    fct->set(decl, body);

    return fct;
}

const ScopeStmt* Parser::parseScope() {
    ScopeStmt* scope = new ScopeStmt();
    scope->setPos1(la().pos1());

    expect(Token::L_BRACE, "scope-statement");

    Stmts& stmts = scope->stmts_;

    while (true) {
        if (isExpr()) {
            stmts.push_back(parseStmt());
            continue;
        }

        switch (la()) {
#define IMPALA_KEY_STMT(tok, t_str) case Token:: tok:
#include <impala/tokenlist.h>
            case Token::L_PAREN:
            case Token::L_BRACE:  
            case Token::ID:          stmts.push_back(parseStmt()); continue;

            case Token::END_OF_FILE:
            case Token::R_BRACE:
                scope->setPos2(la().pos2());
                expect(Token::R_BRACE, "scope-statement");
                return scope;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("statement", "statement list");
                scope->setPos2(prevLoc_.pos2());
                lex(); 
                return scope;
        }
    }
}

/*
 * statements
 */

const Stmt* Parser::parseStmt() {
    if (isExpr())
        return parseExprStmt();

    Location loc = la().loc();
    if (accept(Token::ELSE)) {
        error("'else' without matching 'if'", "statement");
        return new ScopeStmt(loc);
    }

    switch (la()) {
        // expression or a decl
        case Token::ID:
            if (la2() == Token::COLON)
                return parseDeclStmt();
            else {
                error("statement", "");
                return new ScopeStmt(loc);
            }

        // other statements
        case Token::IF:        return parseIfElse();
        case Token::WHILE:     return parseWhile();
        case Token::DO:        return parseDoWhile();
        case Token::FOR:       return parseFor();
        case Token::BREAK:     return parseBreak();
        case Token::CONTINUE:  return parseContinue();
        case Token::RETURN:    return parseReturn();
        case Token::L_BRACE:   return parseScope();
        case Token::SEMICOLON: return new ScopeStmt(lex().loc());
        default:               ANYDSL_UNREACHABLE;
    }
}

const ExprStmt* Parser::parseExprStmt() {
    const Expr* expr = parseExpr(); // discard val
    expect(Token::SEMICOLON, "the end of an expression statement");

    return new ExprStmt(expr, prevLoc_.pos2());
}

const DeclStmt* Parser::parseDeclStmt() {
    const Decl* decl = parseDecl();

    // initialization
    const Expr* init = 0;
    if (la() == Token::COLONEQ) {
        Token op = Token(lex().loc(), Token::ASGN);
        init = tryExpr();
    }

    expect(Token::SEMICOLON, "the end of an declaration statement");

    return new DeclStmt(decl, init, prevLoc_.pos2());
}

const Stmt* Parser::parseIfElse() {
    Position pos1 = eat(Token::IF).pos1();
    const Expr* cond = parseCond("if-statement");
    const Stmt* ifStmt = parseStmt();
    const Stmt* elseStmt = accept(Token::ELSE) ? parseStmt() : new ScopeStmt(prevLoc_);

    return new IfElseStmt(pos1, cond, ifStmt, elseStmt);
}

const Stmt* Parser::parseWhile() {
    Position pos1 = eat(Token::WHILE).pos1();
    const Expr* cond = parseCond("while-statement");

    const Loop* oldLoop = curLoop_;
    WhileStmt*  newLoop = new WhileStmt();
    curLoop_ = newLoop;
    const Stmt* body = parseStmt();
    curLoop_ = oldLoop;

    newLoop->set(pos1, cond, body);

    return newLoop;
}

const Stmt* Parser::parseDoWhile() {
    Position pos1 = eat(Token::DO).pos1();

    const Loop* oldLoop = curLoop_;
    DoWhileStmt*  newLoop = new DoWhileStmt();
    curLoop_ = newLoop;
    const Stmt* body = parseStmt();
    curLoop_ = oldLoop;

    expect(Token::WHILE, "do-while-statement");
    const Expr* cond = parseCond("do-while-statement");
    expect(Token::SEMICOLON, "do-while-statement");

    newLoop->set(pos1, body, cond, prevLoc_.pos2());

    return newLoop;
}

const Stmt* Parser::parseFor() {
    Position pos1 = eat(Token::FOR).pos1();
    expect(Token::L_PAREN, "for-statement");

    const Loop* oldLoop = curLoop_;
    ForStmt*  newLoop = new ForStmt();
    curLoop_ = newLoop;

    // clause 1: decl or expr_opt ';'
    if (la2() == Token::COLON)
        newLoop->set(parseDeclStmt());
    else if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
    } else if (isExpr()) {
        newLoop->set(parseExprStmt());
    } else {
        error("expression or delcaration-statement", 
                "first clause in for-statement");
        newLoop->set(new ExprStmt(new EmptyExpr(prevLoc_), prevLoc_.pos2()));
    }

    // clause 2: expr_opt ';'
    const Expr* cond;
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
        // but create true cond
        cond = new Literal(prevLoc_, Literal::LIT_bool, Box(true));
    } else if (isExpr()) {
        cond = parseExpr();
        expect(Token::SEMICOLON, "second clause in for-statement");
    } else {
        error("expression or nothing", 
                "second clause in for-statement");
        cond = new Literal(prevLoc_, Literal::LIT_bool, Box(true));
    }

    const Expr* inc;
    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
        inc = new EmptyExpr(prevLoc_);
    } else if (isExpr()) {
        inc = parseExpr();
        expect(Token::R_PAREN, "for-statement");
    } else {
        error("expression or nothing",
                "third clause in for-statement");
        inc = new EmptyExpr(prevLoc_);
    }

    const Stmt* body = parseStmt();
    curLoop_ = oldLoop;

    newLoop->set(pos1, cond, inc, body);

    return newLoop;
}

const Stmt* Parser::parseBreak() {
    Position pos1 = eat(Token::BREAK).pos1();
    expect(Token::SEMICOLON, "break-statement");

    return new BreakStmt(pos1, prevLoc_.pos2(), curLoop_);
}

const Stmt* Parser::parseContinue() {
    Position pos1 = eat(Token::CONTINUE).pos1();
    expect(Token::SEMICOLON, "continue-statement");

    return new ContinueStmt(pos1, prevLoc_.pos2(), curLoop_);
}

const Stmt* Parser::parseReturn() {
    Position pos1 = eat(Token::RETURN).pos1();

    const Expr* expr = 0;

    if (la() != Token::SEMICOLON) {
        if (isExpr()) {
            expr = parseExpr();
            expect(Token::SEMICOLON, "return-statement");
        }
    } else {
        expr = new EmptyExpr(la().loc());
        eat(Token::SEMICOLON);
    }

    return new ReturnStmt(pos1, expr, curFct_, prevLoc_.pos2());
}

const Expr* Parser::parseCond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    const Expr* cond = tryExpr();
    expect(Token::R_PAREN, "condition in " + what);

    return cond;
}

bool Parser::isExpr() {
    // identifier without a succeeding colon
    if (la() == Token::ID && la2() != Token::COLON)
        return true;

    switch (la()) {
#define IMPALA_PREFIX(tok, t_str, r) case Token:: tok:
#define IMPALA_KEY_EXPR(tok, t_str)  case Token:: tok:
#define IMPALA_LIT(itype, atype)     case Token:: LIT_##itype:
//#define IMPALA_TYPE(itype, atype)    case Token:: TYPE_ ## itype:
#include <impala/tokenlist.h>
        case Token::L_PAREN:
            return true;
        default:
            return false;
    }
}

const Expr* Parser::tryExpr() {
    return isExpr() ? parseExpr() : new EmptyExpr(prevLoc_);
}

/*
 * expressions
 */

const Expr* Parser::parseExpr(Prec prec) {
    const Expr* lhs = la().isPrefix() ? parsePrefixExpr() : parsePrimaryExpr();

    while (true) {
        /*
         * (aval  op  LA) op ...  on break  (current prec > aval prec of LA)  -->  reduce
         *  aval  op (LA  op ...) otherwise                                   -->  shift
         */

        if (la().isInfix()) {
            if (prec > PrecTable::infix_l[la()])
                break;

            lhs = parseInfixExpr(lhs);
        } else if ( la().isPostfix() ) {
            if (prec > PrecTable::postfix_l[la()])
                break;

            lhs = parsePostfixExpr(lhs);
        } else
            break;
    }

    return lhs;
}

const Expr* Parser::parsePrefixExpr() {
    Token op = lex();
    const Expr* rhs = parseExpr(PrecTable::prefix_r[op]);

    return new PrefixExpr(op.pos1(), (PrefixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parseInfixExpr(const Expr* lhs) {
    Token op = lex();
    const Expr* rhs = parseExpr(PrecTable::infix_r[op]);

    return new InfixExpr(lhs, (InfixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parsePostfixExpr(const Expr* lhs) {
    if (accept(Token::L_PAREN)) {
        Call* call = new Call(lhs);
        PARSE_COMMA_LIST
        (
            call->appendArg(tryExpr()),
            Token::R_PAREN,
            "arguments of a function call"
        )
        call->setLoc(prevLoc_.pos2());

        return call;
    } else {
        assert(la() == Token::INC || la() == Token::DEC);
        Token op = lex();
        return new PostfixExpr(lhs, (PostfixExpr::Kind) op.kind(), op.pos2());
    }
}

const Expr* Parser::parsePrimaryExpr() {
    switch (la() ) {
        case Token::L_PAREN: {
            eat(Token::L_PAREN);
            const Expr* expr = parseExpr();
            expect(Token::R_PAREN, "primary expression");
            return expr;
        }
        case Token::ID:     return new Id(lex());
        //case Token::LAMBDA: return parseLambda();
        //case Token::PI:            return parsePi();
        //case Token::SIGMA:         return parseSigma();
        //case Token::MEMORY:        return parseMemory();

#define IMPALA_LIT(itype, atype) case Token:: LIT_##itype:
#include <impala/tokenlist.h>
        case Token::TRUE:
        case Token::FALSE:
            return parseLiteral();

//#define IMPALA_TYPE(itype, atype) case Token:: TYPE_ ## itype:
//#include <impala/tokenlist.h>
            //return Value(parseType());

        default: {
            error("expression", "primary expression");
            return new EmptyExpr(prevLoc_);
        }
    }
}

const Expr* Parser::parseLiteral() {
    Literal::Kind kind;
    Box value;

    switch (la()) {
        case Token::TRUE:  return new Literal(lex().loc(), Literal::LIT_bool, Box(true));
        case Token::FALSE: return new Literal(lex().loc(), Literal::LIT_bool, Box(false));
#define IMPALA_LIT(itype, atype) \
        case Token::LIT_##itype: { \
            kind = Literal::LIT_##itype; \
            Box value = la().box(); \
            return new Literal(lex().loc(), kind, value); \
        }
#include "impala/tokenlist.h"
        default: ANYDSL_UNREACHABLE;
    }
}

const Expr* Parser::parseLambda() {
#if 0
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
#endif
    return 0;
}

} // namespace impala
