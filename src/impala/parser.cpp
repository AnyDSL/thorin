#include "impala/parser.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "anydsl2/util/array.h"
#include "anydsl2/util/assert.h"

#include "impala/ast.h"
#include "impala/lexer.h"
#include "impala/prec.h"
#include "impala/type.h"

using namespace anydsl2;

namespace impala {

class Parser {
public:

    /*
     * constructor
     */

    Parser(World& world, std::istream& stream, const std::string& filename);

    /*
     * helpers
     */

    const Token& la(size_t i) const { return lookahead_[i]; }
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
    std::ostream& sema_error();
    bool result() const { return result_; }

    // misc
    Token try_id(const std::string& what);
    const Type* try_type(const std::string& what);
    const Expr* try_expr(const std::string& what);
    bool is_expr();
    bool is_type(size_t lookahead = 0);
    const Prg* parse();
    const Type* parse_type();
    const Decl* parse_decl();
    void parse_globals();
    void parse_lambda(Lambda* lambda);
    const Fct* parse_fct();

    // expressions
    const Expr* parse_expr(Prec prec);
    const Expr* parse_expr() { return parse_expr(BOTTOM); }
    const Expr* parse_prefix_expr();
    const Expr* parse_infix_expr(const Expr* lhs);
    const Expr* parse_postfix_expr(const Expr* lhs);
    const Expr* parse_primary_expr();
    const Expr* parse_literal();
    const Expr* parse_tuple();
    const Expr* parse_lambda_expr();

    // statements
    const Stmt* parse_stmt();
    const ExprStmt* parse_expr_stmt();
    const DeclStmt* parse_decl_stmt();
    const Stmt* parse_if_else();
    const Stmt* parse_while();
    const Stmt* parse_do_while();
    const Stmt* parse_for();
    const Stmt* parse_break();
    const Stmt* parse_continue();
    const Stmt* parse_return();

    /// { Stmt* }
    const ScopeStmt* parse_scope();

    /// helper for condition in if/while/do-while
    const Expr* parse_cond(const std::string& what);

private:

    int next_id();

    /// Consume next Token in input stream, fill look-ahead buffer, return consumed Token.
    Token lex();

    /*
     * data
     */

    typedef boost::unordered_map<Symbol, const Generic*> Generics;
    Generics generics_;
    World& world;
    Lexer lexer_;       ///< invoked in order to get next token
    Token lookahead_[2];///< LL(2) look ahead
    const Loop* cur_loop_;
    const Lambda* cur_lambda_;
    Prg* prg_;
    int counter_;
    anydsl2::Location prev_loc_;
    bool result_;
};

//------------------------------------------------------------------------------

const Prg* parse(World& world, std::istream& i, const std::string& filename, bool& result) {
    Parser p(world, i, filename);
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
 * constructor
 */

Parser::Parser(World& world, std::istream& stream, const std::string& filename)
    : world(world)
    , lexer_(stream, filename)
    , cur_loop_(0)
    , cur_lambda_(0)
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
    prev_loc_ = result.loc();       // remember previous location

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
    std::ostream& os = la().error() << "expected " << what << ", got '" << la() << "'";
    if (!context.empty())
        os << " while parsing " << context;
    os << "\n";
}

std::ostream& Parser::sema_error() {
    result_ = false;
    return la().error();
}

int Parser::next_id() {
    int id = counter_++;
    counter_ %= 0x100;

    return id;
}

/*
 * misc
 */

const Prg* Parser::parse() {
    parse_globals();
    return prg_;
}

Token Parser::try_id(const std::string& what) {
    Token name;
    if (la() == Token::ID)
        name = lex();
    else {
        error("identifier", what);
        name = Token(la().loc(), make_symbol("error-name", next_id()).str());
    }

    return name;
}

const Type* Parser::parse_type() {
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token::TYPE_##itype: lex(); return world.type_##atype();
#include "impala/tokenlist.h"

        case Token::TYPE_int:   lex(); return world.type_u32();
        case Token::TYPE_void:  lex(); return world.unit();
        case Token::TYPE_noret: lex(); return world.noret();
        case Token::PI: {
            lex();
            std::vector<const Type*> elems;
            expect(Token::L_PAREN, "element types of pi");
            PARSE_COMMA_LIST
            (
                elems.push_back(try_type("element type of pi")),
                Token::R_PAREN,
                "closing parenthesis of pi type"
            )

            if (accept(Token::ARROW))
                elems.push_back(world.pi1(parse_type()));

            return world.pi(elems);
        }
        case Token::L_TUPLE: {
            lex();
            std::vector<const Type*> elems;
            PARSE_COMMA_LIST
            (
                elems.push_back(try_type("element of tuple type")),
                Token::R_PAREN,
                "closing parenthesis of tuple type"
            )

            return world.sigma(elems);
        }
        case Token::ID: {
            Symbol id = lex().symbol();
            Generics::iterator i = generics_.find(id);
            assert(i != generics_.end());
            return i->second;
        }
        default: ANYDSL2_UNREACHABLE; // TODO
    }
}

const Decl* Parser::parse_decl() {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
    const Type* type = try_type("declaration");

    return new Decl(tok, type, prev_loc_.pos2());
}

void Parser::parse_globals() {
    while (true) {
        switch (la()) {
            case Token::DEF:         prg_->fcts_.push_back(parse_fct()); continue;
            case Token::END_OF_FILE: return;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("global", "global list");
                lex(); 
        }
    }
}

void Parser::parse_lambda(Lambda* lambda) {
    IMPALA_PUSH(cur_lambda_, lambda);

    std::vector<const Type*> arg_types;

#if 0
    if (accept(Token::LT))
        PARSE_COMMA_LIST
        (
            {
                Symbol id = try_id("generic identifier").symbol();
                const Generic* generic = world.generic(id, cur_lambda_);
                Generics::iterator i = generics_.find(id);
                if (i == generics_.end()) {
                    lambda->generics_.push_back(generic);
                    generics_[id] = generic;
                } else
                    sema_error() << "generic " << id << " defined twice\n";
            },
            Token::GT,
            "generics list"
        )
#endif

    expect(Token::L_PAREN, "function head");
    PARSE_COMMA_LIST
    (
        {
            const Decl* param = parse_decl();
            lambda->params_.push_back(param);
            arg_types.push_back(param->type());
        },
        Token::R_PAREN,
        "parameter list"
    )

    // return-continuation
    if (accept(Token::ARROW))
        arg_types.push_back(world.pi1(try_type("return type of function definition")));

    const Pi* pi = world.pi(arg_types);
    const ScopeStmt* body = parse_scope();

    lambda->set(pi, body);
    generics_.clear();

    IMPALA_POP(cur_lambda_);
}

const Fct* Parser::parse_fct() {
    Position pos1 = eat(Token::DEF).pos1();
    Token id = try_id("function identifier");

    Fct* fct = new Fct();
    parse_lambda(&fct->lambda_);

    const Decl* decl = new Decl(id, fct->lambda().pi(), prev_loc_.pos2());

    fct->set(decl);

    return fct;
}

const ScopeStmt* Parser::parse_scope() {
    ScopeStmt* scope = new ScopeStmt();
    scope->set_pos1(la().pos1());

    expect(Token::L_BRACE, "scope-statement");

    Stmts& stmts = scope->stmts_;

    while (true) {
        if (is_expr()) {
            stmts.push_back(parse_stmt());
            continue;
        }

        switch (la()) {
#define IMPALA_KEY_STMT(tok, t_str) case Token:: tok:
#include <impala/tokenlist.h>
            case Token::L_PAREN:
            case Token::L_BRACE:  
            case Token::ID:          stmts.push_back(parse_stmt()); continue;

            case Token::END_OF_FILE:
            case Token::R_BRACE:
                scope->set_pos2(la().pos2());
                expect(Token::R_BRACE, "scope-statement");
                return scope;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("statement", "statement list");
                scope->set_pos2(prev_loc_.pos2());
                lex(); 
                return scope;
        }
    }
}

/*
 * statements
 */

const Stmt* Parser::parse_stmt() {
    if (is_expr())
        return parse_expr_stmt();

    Location loc = la().loc();
    if (accept(Token::ELSE)) {
        error("'else' without matching 'if'", "statement");
        return new ScopeStmt(loc);
    }

    switch (la()) {
        // expression or a decl
        case Token::ID:
            if (la2() == Token::COLON)
                return parse_decl_stmt();
            else {
                error("statement", "");
                return new ScopeStmt(loc);
            }

        // other statements
        case Token::IF:        return parse_if_else();
        case Token::WHILE:     return parse_while();
        case Token::DO:        return parse_do_while();
        case Token::FOR:       return parse_for();
        case Token::BREAK:     return parse_break();
        case Token::CONTINUE:  return parse_continue();
        case Token::RETURN:    return parse_return();
        case Token::L_BRACE:   return parse_scope();
        case Token::SEMICOLON: return new ScopeStmt(lex().loc());
        default:               ANYDSL2_UNREACHABLE;
    }
}

const ExprStmt* Parser::parse_expr_stmt() {
    const Expr* expr = parse_expr(); // discard val
    expect(Token::SEMICOLON, "the end of an expression statement");

    return new ExprStmt(expr, prev_loc_.pos2());
}

const DeclStmt* Parser::parse_decl_stmt() {
    const Decl* decl = parse_decl();

    // initialization
    const Expr* init = 0;
    if (la() == Token::ASGN) {
        Token op = Token(lex().loc(), Token::ASGN);
        init = try_expr("right-hand side of an initialization");
    }

    expect(Token::SEMICOLON, "the end of an declaration statement");

    return new DeclStmt(decl, init, prev_loc_.pos2());
}

const Stmt* Parser::parse_if_else() {
    Position pos1 = eat(Token::IF).pos1();
    const Expr* cond = parse_cond("if-statement");
    const Stmt* ifStmt = parse_stmt();
    const Stmt* elseStmt = accept(Token::ELSE) ? parse_stmt() : new ScopeStmt(prev_loc_);

    return new IfElseStmt(pos1, cond, ifStmt, elseStmt);
}

const Stmt* Parser::parse_while() {
    Position pos1 = eat(Token::WHILE).pos1();
    const Expr* cond = parse_cond("while-statement");

    WhileStmt*  new_loop = new WhileStmt();
    IMPALA_PUSH(cur_loop_, new_loop);
        const Stmt* body = parse_stmt();
    IMPALA_POP(cur_loop_);

    new_loop->set(pos1, cond, body);

    return new_loop;
}

const Stmt* Parser::parse_do_while() {
    Position pos1 = eat(Token::DO).pos1();

    DoWhileStmt* new_loop = new DoWhileStmt();
    IMPALA_PUSH(cur_loop_, new_loop);
        const Stmt* body = parse_stmt();
    IMPALA_POP(cur_loop_);

    expect(Token::WHILE, "do-while-statement");
    const Expr* cond = parse_cond("do-while-statement");
    expect(Token::SEMICOLON, "do-while-statement");

    new_loop->set(pos1, body, cond, prev_loc_.pos2());

    return new_loop;
}

const Stmt* Parser::parse_for() {
    Position pos1 = eat(Token::FOR).pos1();
    expect(Token::L_PAREN, "for-statement");

    ForStmt*  new_loop = new ForStmt();
    IMPALA_PUSH(cur_loop_, new_loop);

    // clause 1: decl or expr_opt ';'
    if (la2() == Token::COLON)
        new_loop->set(parse_decl_stmt());
    else if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
    } else if (is_expr()) {
        new_loop->set(parse_expr_stmt());
    } else {
        error("expression or delcaration-statement", 
                "first clause in for-statement");
        new_loop->set(new ExprStmt(new EmptyExpr(prev_loc_), prev_loc_.pos2()));
    }

    // clause 2: expr_opt ';'
    const Expr* cond;
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
        // but create true cond
        cond = new Literal(prev_loc_, Literal::LIT_bool, Box(true));
    } else if (is_expr()) {
        cond = parse_expr();
        expect(Token::SEMICOLON, "second clause in for-statement");
    } else {
        error("expression or nothing", 
                "second clause in for-statement");
        cond = new Literal(prev_loc_, Literal::LIT_bool, Box(true));
    }

    const Expr* inc;
    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
        inc = new EmptyExpr(prev_loc_);
    } else if (is_expr()) {
        inc = parse_expr();
        expect(Token::R_PAREN, "for-statement");
    } else {
        error("expression or nothing",
                "third clause in for-statement");
        inc = new EmptyExpr(prev_loc_);
    }

    const Stmt* body = parse_stmt();
    new_loop->set(pos1, cond, inc, body);
    IMPALA_POP(cur_loop_);

    return new_loop;
}

const Stmt* Parser::parse_break() {
    Position pos1 = eat(Token::BREAK).pos1();
    expect(Token::SEMICOLON, "break-statement");

    return new BreakStmt(pos1, prev_loc_.pos2(), cur_loop_);
}

const Stmt* Parser::parse_continue() {
    Position pos1 = eat(Token::CONTINUE).pos1();
    expect(Token::SEMICOLON, "continue-statement");

    return new ContinueStmt(pos1, prev_loc_.pos2(), cur_loop_);
}

const Stmt* Parser::parse_return() {
    Position pos1 = eat(Token::RETURN).pos1();
    const Expr* expr;

    if (la() != Token::SEMICOLON) {
        expr = try_expr("return-statement");
        expect(Token::SEMICOLON, "return-statement");
    } else {
        expr = new EmptyExpr(la().loc());
        eat(Token::SEMICOLON);
    }

    return new ReturnStmt(pos1, expr, cur_lambda_, prev_loc_.pos2());
}

const Expr* Parser::parse_cond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    const Expr* cond = try_expr("condition in " + what);
    expect(Token::R_PAREN, "condition in " + what);

    return cond;
}

bool Parser::is_expr() {
    // identifier without a succeeding colon which is not a generic
    if (la() == Token::ID 
            && la2() != Token::COLON 
            && generics_.find(la().symbol()) == generics_.end())
        return true;

    switch (la()) {
#define IMPALA_PREFIX(tok, t_str, r) case Token:: tok:
#define IMPALA_KEY_EXPR(tok, t_str)  case Token:: tok:
#define IMPALA_LIT(itype, atype)     case Token:: LIT_##itype:
#include <impala/tokenlist.h>
        case Token::L_PAREN:
        case Token::L_TUPLE:
            return true;
        default:
            return false;
    }
}

bool Parser::is_type(size_t lookahead) {
    // generics
    if (la(lookahead) == Token::ID
            && generics_.find(la().symbol()) != generics_.end())
        return true;

    switch (la(lookahead)) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_##itype:
#include <impala/tokenlist.h>
        case Token::TYPE_int:
        case Token::TYPE_uint:
        case Token::TYPE_void:
        case Token::TYPE_noret:
        case Token::PI:
            return true;
        case Token::L_TUPLE:
            assert(lookahead == 0);
            return is_type(1);
        default:
            return false;
    }
}

const Expr* Parser::try_expr(const std::string& what) {
    const Expr* expr;
    if (is_expr())
        expr = parse_expr();
    else {
        error("expression", what);
        expr = new EmptyExpr(prev_loc_);
    }

    return expr;
}

const Type* Parser::try_type(const std::string& what) {
    const Type* type;
    if (is_type())
        type = parse_type();
    else {
        error("type", what);
        type = world.type_error();
    }

    return type;
}

/*
 * expressions
 */

const Expr* Parser::parse_expr(Prec prec) {
    const Expr* lhs = la().is_prefix() ? parse_prefix_expr() : parse_primary_expr();

    while (true) {
        /*
         * (aval  op  LA) op ...  on break  (current prec > aval prec of LA)  -->  reduce
         *  aval  op (LA  op ...) otherwise                                   -->  shift
         */

        if (la().is_infix()) {
            if (prec > PrecTable::infix_l[la()])
                break;

            lhs = parse_infix_expr(lhs);
        } else if ( la().is_postfix() ) {
            if (prec > PrecTable::postfix_l[la()])
                break;

            lhs = parse_postfix_expr(lhs);
        } else
            break;
    }

    return lhs;
}

const Expr* Parser::parse_prefix_expr() {
    Token op = lex();
    const Expr* rhs = parse_expr(PrecTable::prefix_r[op]);

    return new PrefixExpr(op.pos1(), (PrefixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_infix_expr(const Expr* lhs) {
    Token op = lex();
    const Expr* rhs = parse_expr(PrecTable::infix_r[op]);

    return new InfixExpr(lhs, (InfixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_postfix_expr(const Expr* lhs) {
    if (accept(Token::L_PAREN)) {
        Call* call = new Call(lhs);
        PARSE_COMMA_LIST
        (
            call->append_arg(try_expr("argument of function call")),
            Token::R_PAREN,
            "arguments of a function call"
        )
        call->set_pos2(prev_loc_.pos2());

        return call;
    } else if (accept(Token::L_BRACKET)) {
        Position pos1 = prev_loc_.pos1();
        const Expr* index = try_expr("indexing expression");
        expect(Token::R_BRACKET, "index expression");
        return new IndexExpr(pos1, lhs, index, prev_loc_.pos2());
    } else {
        assert(la() == Token::INC || la() == Token::DEC);
        Token op = lex();
        return new PostfixExpr(lhs, (PostfixExpr::Kind) op.kind(), op.pos2());
    }
}

const Expr* Parser::parse_primary_expr() {
    switch (la() ) {
#define IMPALA_LIT(itype, atype) \
        case Token::LIT_##itype:
#include <impala/tokenlist.h>
        case Token::TRUE:
        case Token::FALSE:      return parse_literal();
        case Token::ID: {
            assert(generics_.find(la().symbol()) == generics_.end());
            return new Id(lex());
        }
        case Token::L_PAREN: {
            eat(Token::L_PAREN);
            const Expr* expr = try_expr("primary expression");
            expect(Token::R_PAREN, "primary expression");
            return expr;
        }
        case Token::L_TUPLE:    return parse_tuple();
        case Token::LAMBDA:     return parse_lambda_expr();
        default:                ANYDSL2_UNREACHABLE;
    }
}

const Expr* Parser::parse_literal() {
    Literal::Kind kind;
    Box box;

    switch (la()) {
        case Token::TRUE:  return new Literal(lex().loc(), Literal::LIT_bool, Box(true));
        case Token::FALSE: return new Literal(lex().loc(), Literal::LIT_bool, Box(false));
#define IMPALA_LIT(itype, atype) \
        case Token::LIT_##itype: { \
            kind = Literal::LIT_##itype; \
            Box box = la().box(); \
            return new Literal(lex().loc(), kind, box); \
        }
#include "impala/tokenlist.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const Expr* Parser::parse_tuple() {
    Tuple* tuple = new Tuple(eat(Token::L_TUPLE).pos1());

    PARSE_COMMA_LIST
    (
        tuple->ops_.push_back(try_expr("tuple element")),
        Token::R_PAREN,
        "closing parenthesis of tuple"
    )

    tuple->set_pos2(prev_loc_.pos2());

    return tuple;
}

const Expr* Parser::parse_lambda_expr() {
    Position pos1 = eat(Token::LAMBDA).pos1();
    LambdaExpr* lambda_expr = new LambdaExpr();
    parse_lambda(&lambda_expr->lambda_);

    return lambda_expr;
}

} // namespace impala
