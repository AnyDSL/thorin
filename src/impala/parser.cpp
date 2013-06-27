#include "impala/parser.h"

#include <algorithm>
#include <sstream>

#include "anydsl2/util/array.h"
#include "anydsl2/util/assert.h"
#include "anydsl2/util/push.h"

#include "impala/ast.h"
#include "impala/lexer.h"
#include "impala/prec.h"
#include "impala/type.h"

#define PARSE_COMMA_LIST(stmnt, delimiter, context) { \
        if (la() != delimiter) { \
            do { stmnt; } \
            while ( accept(Token::COMMA) ); \
        } \
        expect(delimiter, context); \
    }

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

    const Token& la(size_t i) const { return lookahead[i]; }
    const Token& la () const { return lookahead[0]; }
    const Token& la2() const { return lookahead[1]; }

#ifdef NDEBUG
    Token eat(TokenKind /*what*/) { return lex(); }
#else
    Token eat(TokenKind what) { assert(what == la() && "internal parser error"); return lex(); }
#endif

    bool accept(TokenKind tok);

    bool expect(TokenKind tok, const std::string& context);
    void error(const std::string& what, const std::string& context);
    std::ostream& sema_error(Token token);
    bool result() const { return result_; }

    // misc
    Token try_id(const std::string& what);
    const Type* try_type(const std::string& what);
    const Expr* try_expr(Prec prec, const std::string& what);
    const Expr* try_expr(const std::string& what) { return try_expr(BOTTOM, what); }
    const Stmt* try_stmt(const std::string& what);
    bool is_type(size_t lookahead = 0);
    bool is_expr();
    bool is_stmt();
    const Prg* parse();
    void parse_generic_list();
    const Type* parse_type();
    const Type* parse_compound_type();
    const Type* parse_return_type();
    const VarDecl* parse_var_decl();
    void parse_globals();
    const Proto* parse_proto();
    void parse_fun(Fun* fun);
    const NamedFun* parse_named_fun();

    // expressions
    const Expr* parse_expr(Prec prec);
    const Expr* parse_expr() { return parse_expr(BOTTOM); }
    const Expr* parse_prefix_expr();
    const Expr* parse_infix_expr(const Expr* lhs);
    const Expr* parse_postfix_expr(const Expr* lhs);
    const Expr* parse_primary_expr();
    const Expr* parse_literal();
    const Expr* parse_tuple();
    const Expr* parse_fun_expr();

    // statements
    const Stmt* parse_stmt();
    const ExprStmt* parse_expr_stmt();
    const DeclStmt* parse_decl_stmt();
    const Stmt* parse_if_else();
    const Stmt* parse_while();
    const Stmt* parse_do_while();
    const Stmt* parse_for();
    const Stmt* parse_foreach();
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

    typedef boost::unordered_map<Symbol, size_t> Symbol2Handle;
    class Generics {
    public:

        Generics(Generics* parent, GenericBuilder& builder)
            : parent_(parent)
            , builder(builder)
            , counter(0)
        {}
        ~Generics() {
            for (; counter; --counter)
                builder.pop();
        }

        Generics* parent() const { return parent_; }

        const Generic* lookup(Symbol symbol) {
            Symbol2Handle::iterator i = symbol2handle_.find(symbol);
            if (i != symbol2handle_.end()) 
                return builder.use(i->second);

            if (parent_)
                return parent_->lookup(symbol);

            return 0;
        }

        void insert(Symbol symbol) { 
            ++counter;
            symbol2handle_[symbol] = builder.new_def();
        }

    private:

        Generics* parent_;
        GenericBuilder& builder;
        int counter;
        Symbol2Handle symbol2handle_;
    };

    const Generic* generic_lookup(Symbol symbol) {
        return cur_generics ? cur_generics->lookup(symbol) : 0;
    }

    void generic_insert(Token token) {
        Symbol symbol = token.symbol();
        assert(cur_generics);
        if (cur_generics->lookup(symbol))
            sema_error(token) << "generic '" << symbol << "' defined twice\n";
        else
            cur_generics->insert(symbol);
    }

    World& world;
    Lexer lexer;       ///< invoked in order to get next token
    Token lookahead[2];///< LL(2) look ahead
    const Loop* cur_loop;
    const Fun* cur_fun;
    Generics* cur_generics;
    size_t cur_var_handle;
    size_t generic_counter;
    Prg* prg;
    int counter;
    anydsl2::Location prev_loc;
    bool result_;
    GenericBuilder builder;
};

//------------------------------------------------------------------------------

const Prg* parse(World& world, std::istream& i, const std::string& filename, bool& result) {
    Parser p(world, i, filename);
    const Prg* prg = p.parse();
    result = p.result();

    return prg;
}

//------------------------------------------------------------------------------

/*
 * constructor
 */

Parser::Parser(World& world, std::istream& stream, const std::string& filename)
    : world(world)
    , lexer(stream, filename)
    , cur_loop(0)
    , cur_fun(0)
    , cur_generics(0)
    , cur_var_handle(2) // reserve 0 for conditionals, 1 for mem
    , generic_counter(0)
    , prg(new Prg())
    , counter(0)
    , result_(true)
    , builder(world)
{
    // init 2 lookahead
    lookahead[0] = lexer.lex();
    lookahead[1] = lexer.lex();
}

/*
 * helpers
 */

Token Parser::lex() {
    Token result  = lookahead[0]; // remember result
    lookahead[0] = lookahead[1];  // copy over LA2 to LA1
    lookahead[1] = lexer.lex();   // fill new LA2
    prev_loc = result.loc();      // remember previous location

    return result;
}

bool Parser::accept(TokenKind type) {
    if (type != la())
        return false;
    lex();
    return true;
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

std::ostream& Parser::sema_error(Token token) {
    result_ = false;
    return token.error();
}

int Parser::next_id() {
    int id = counter++;
    counter %= 0x100;

    return id;
}

/*
 * misc
 */

const Prg* Parser::parse() {
    parse_globals();
    return prg;
}

Token Parser::try_id(const std::string& what) {
    Token name;
    if (la() == Token::ID)
        name = lex();
    else {
        error("identifier", what);
        name = Token(la().loc(), "<error_id>");
    }

    return name;
}

const Type* Parser::parse_type() {
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token::TYPE_##itype: lex(); return world.type_##atype();
#include "impala/tokenlist.h"

        case Token::TYPE_int:   lex(); return world.type_u32();
        case Token::TYPE_void:  lex(); return world.type_void();
        case Token::TYPE_noret: lex(); return world.noret();
        case Token::PI:
        case Token::SIGMA:             return parse_compound_type();
        case Token::ID:
            return cur_generics->lookup(lex().symbol());
        default: ANYDSL2_UNREACHABLE;
    }
}

const Type* Parser::parse_compound_type() {
    bool pi = lex().kind() == Token::PI;

    Generics generics(cur_generics, builder);
    cur_generics = &generics;
    parse_generic_list();

    std::vector<const Type*> elems;
    const char* error_str = pi ? "element types of pi" : "element types of sigma";
    expect(Token::L_PAREN, error_str);
    PARSE_COMMA_LIST
    (
        elems.push_back(try_type(error_str)),
        Token::R_PAREN,
        pi ?  "closing parenthesis of pi type" : "closing parenthesis of sigma type"
    )

    if (pi && accept(Token::ARROW))
        elems.push_back(parse_return_type());

    cur_generics = generics.parent();

    if (pi) 
        return world.pi(elems);
    return world.sigma(elems);
}

const Type* Parser::parse_return_type() {
    const Type* ret_type = try_type("return type");
    if (ret_type->isa<Void>())
        return world.pi0();
    else
        return world.pi1(ret_type);
}

const VarDecl* Parser::parse_var_decl() {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
    const Type* type = try_type("declaration");

    return new VarDecl(cur_var_handle++, tok, type, prev_loc.pos2());
}

void Parser::parse_globals() {
    while (true) {
        switch (la()) {
            case Token::SEMICOLON:   lex(); continue; // ignore semicolon in global list
            case Token::DEF:         prg->globals_.push_back(parse_named_fun()); continue;
            case Token::EXTERN:      prg->globals_.push_back(parse_proto()); continue;
            case Token::END_OF_FILE: return;

            // consume token nobody wants to have in order to prevent infinite loop
            default:
                error("global", "global list");
                lex(); 
        }
    }
}

const Proto* Parser::parse_proto() {
    Position pos1 = la().pos1();
    eat(Token::EXTERN);
    Proto* proto = new Proto(try_id("prototype").symbol());
    std::vector<const Type*> types;
    expect(Token::L_PAREN, "prototype");
    PARSE_COMMA_LIST
    (
        types.push_back(try_type("type list of prototype function")),
        Token::R_PAREN,
        "type list of prototype"
    )

    expect(Token::ARROW, "prototype");

    const Type* ret_type = try_type("return type of prototype");
    if (ret_type->isa<Void>())
        types.push_back(world.pi0());
    else
        types.push_back(world.pi1(ret_type));

    proto->type_ = world.pi(types);
    proto->set_loc(pos1, prev_loc.pos2());

    return proto;
}

void Parser::parse_generic_list() {
    if (accept(Token::LT)) {
        PARSE_COMMA_LIST
        (
            generic_insert(try_id("generic identifier")),
            Token::GT,
            "generics list"
        )
    }
}

void Parser::parse_fun(Fun* fun) {
    Push<const Fun*> push1(cur_fun, fun);
    Push<size_t>     push2(cur_var_handle, cur_var_handle);

    Generics generics(cur_generics, builder);
    cur_generics = &generics;
    parse_generic_list();

    std::vector<const Type*> arg_types;
    expect(Token::L_PAREN, "function head");
    PARSE_COMMA_LIST
    (
        {
            const VarDecl* param = parse_var_decl();
            fun->params_.push_back(param);
            arg_types.push_back(param->type());
        },
        Token::R_PAREN,
        "parameter list"
    )

    // return-continuation
    if (accept(Token::ARROW)) {
        Position pos1 = prev_loc.pos1();
        arg_types.push_back(parse_return_type());
        Position pos2 = prev_loc.pos2();
        fun->params_.push_back(new VarDecl(cur_var_handle++, Token(pos1, "return"), arg_types.back(), pos2));
    }

    const Pi* pi = world.pi(arg_types);
    const ScopeStmt* body = parse_scope();
    fun->fun_set(pi, body);

    cur_generics = generics.parent();
}

const NamedFun* Parser::parse_named_fun() {
    Position pos1 = eat(Token::DEF).pos1();
    bool ext = accept(Token::EXTERN);
    Token id = try_id("function identifier");

    NamedFun* f = new NamedFun(ext);
    parse_fun(f);
    f->set(id, f->pi(), prev_loc.pos2());

    return f;
}

const ScopeStmt* Parser::parse_scope() {
    ScopeStmt* scope = new ScopeStmt();
    scope->set_pos1(la().pos1());

    expect(Token::L_BRACE, "scope statement");

    Stmts& stmts = scope->stmts_;
    NamedFuns& named_funs = scope->named_funs_;

    while (true) {
        if (la() == Token::DEF) {
            const NamedFun* named_fun = parse_named_fun();
            const NamedFunStmt* named_fun_stmt = new NamedFunStmt(named_fun);
            stmts.push_back(named_fun_stmt);
            named_funs.push_back(named_fun);
        } else if (is_stmt())
            stmts.push_back(parse_stmt());
        else
            break;
    }

    expect(Token::R_BRACE, "scope statement");

    return scope;
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
        case Token::FOREACH:   return parse_foreach();
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

    return new ExprStmt(expr, prev_loc.pos2());
}

const DeclStmt* Parser::parse_decl_stmt() {
    const VarDecl* var_decl = parse_var_decl();

    // initialization
    const Expr* init = 0;
    if (la() == Token::ASGN) {
        Token op = Token(lex().loc(), Token::ASGN);
        init = try_expr("right-hand side of an initialization");
    }

    expect(Token::SEMICOLON, "the end of an declaration statement");

    return new DeclStmt(var_decl, init, prev_loc.pos2());
}

const Stmt* Parser::parse_if_else() {
    Position pos1 = eat(Token::IF).pos1();
    const Expr* cond = parse_cond("if statement");
    const Stmt* ifStmt = try_stmt("if clause");
    const Stmt* elseStmt = accept(Token::ELSE) ? try_stmt("else clause") : new ScopeStmt(prev_loc);

    return new IfElseStmt(pos1, cond, ifStmt, elseStmt);
}

const Stmt* Parser::parse_while() {
    Position pos1 = eat(Token::WHILE).pos1();
    const Expr* cond = parse_cond("while statement");

    ForStmt* loop = new ForStmt();
    Push<const Loop*> push(cur_loop, loop);
    const Stmt* body = try_stmt("loop body");

    loop->set(pos1, cond, new EmptyExpr(pos1), body);
    loop->set_empty_init(pos1);

    return loop;
}

const Stmt* Parser::parse_do_while() {
    Position pos1 = eat(Token::DO).pos1();

    DoWhileStmt* loop = new DoWhileStmt();
    Push<const Loop*> push(cur_loop, loop);
    const Stmt* body = try_stmt("loop body");

    expect(Token::WHILE, "do-while statement");
    const Expr* cond = parse_cond("do-while statement");
    expect(Token::SEMICOLON, "do-while statement");

    loop->set(pos1, body, cond, prev_loc.pos2());

    return loop;
}

const Stmt* Parser::parse_for() {
    Position pos1 = eat(Token::FOR).pos1();
    expect(Token::L_PAREN, "for statement");

    ForStmt*  loop = new ForStmt();
    Push<const Loop*> push(cur_loop, loop);

    // clause 1: decl or expr_opt ';'
    if (la2() == Token::COLON)
        loop->set(parse_decl_stmt());
    else if (is_expr())
        loop->set(parse_expr_stmt());
    else  {
        if (!accept(Token::SEMICOLON))
            error("expression or delcaration statement", "first clause in for statement");
        loop->set_empty_init(prev_loc.pos2());
    }

    // clause 2: expr_opt ';'
    const Expr* cond;
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed
        // but create true cond
        cond = new Literal(prev_loc, Literal::LIT_bool, Box(true));
    } else if (is_expr()) {
        cond = parse_expr();
        expect(Token::SEMICOLON, "second clause in for statement");
    } else {
        error("expression or nothing", 
                "second clause in for statement");
        cond = new Literal(prev_loc, Literal::LIT_bool, Box(true));
    }

    const Expr* inc;
    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
        inc = new EmptyExpr(prev_loc);
    } else if (is_expr()) {
        inc = parse_expr();
        expect(Token::R_PAREN, "for statement");
    } else {
        error("expression or nothing",
                "third clause in for statement");
        inc = new EmptyExpr(prev_loc);
    }

    const Stmt* body = try_stmt("loop body");
    loop->set(pos1, cond, inc, body);

    return loop;
}

const Stmt* Parser::parse_foreach() {
    Position pos1 = eat(Token::FOREACH).pos1();
    expect(Token::L_PAREN, "foreach statement");

    ForeachStmt* foreach = new ForeachStmt();

    if (la2() == Token::COLON) {
        // parse only 'x : int' and no further assignment
        const VarDecl* decl = parse_var_decl();
        foreach->set(decl);
    } else if (is_expr()) {
        const Expr* expr = parse_primary_expr();
        foreach->set(expr);
    } else
        error("expression or delcaration statement", "for-each statement");

    expect(Token::LARROW, "for-each statement");
    const Expr* expr = try_expr("generator call in for-each statement");
    foreach->set(expr->isa<Call>());
    if (!foreach->call())
        error("generator call", " for-each statement");

    expect(Token::R_PAREN, "for-each statement");
    foreach->set(pos1, try_stmt("for-each body"));

    return foreach;
}

const Stmt* Parser::parse_break() {
    Position pos1 = eat(Token::BREAK).pos1();
    expect(Token::SEMICOLON, "break statement");

    return new BreakStmt(pos1, prev_loc.pos2(), cur_loop);
}

const Stmt* Parser::parse_continue() {
    Position pos1 = eat(Token::CONTINUE).pos1();
    expect(Token::SEMICOLON, "continue statement");

    return new ContinueStmt(pos1, prev_loc.pos2(), cur_loop);
}

const Stmt* Parser::parse_return() {
    Position pos1 = eat(Token::RETURN).pos1();
    const Expr* expr;

    if (la() != Token::SEMICOLON) {
        expr = try_expr("return statement");
        expect(Token::SEMICOLON, "return statement");
    } else {
        expr = 0;
        eat(Token::SEMICOLON);
    }

    return new ReturnStmt(pos1, expr, cur_fun, prev_loc.pos2());
}

const Expr* Parser::parse_cond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    const Expr* cond = try_expr("condition in " + what);
    expect(Token::R_PAREN, "condition in " + what);

    return cond;
}

bool Parser::is_type(size_t lookahead) {
    if (la(lookahead) == Token::ID && generic_lookup(la().symbol()))
        return true;

    switch (la(lookahead)) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_##itype:
#include "impala/tokenlist.h"
        case Token::TYPE_int:
        case Token::TYPE_uint:
        case Token::TYPE_void:
        case Token::TYPE_noret:
        case Token::PI:
        case Token::SIGMA:
            return true;
        default:
            return false;
    }
}

bool Parser::is_expr() {
    // identifier without a succeeding colon which is not a generic
    if (la() == Token::ID && la2() != Token::COLON && !generic_lookup(la().symbol()))
        return true;

    switch (la()) {
#define IMPALA_PREFIX(tok, t_str, r) case Token:: tok:
#define IMPALA_KEY_EXPR(tok, t_str)  case Token:: tok:
#define IMPALA_LIT(itype, atype)     case Token:: LIT_##itype:
#include "impala/tokenlist.h"
        case Token::HASH:
        case Token::L_PAREN:
            return true;
        default:
            return false;
    }
}

bool Parser::is_stmt() {
    if (is_expr())
        return true;

    switch (la()) {
#define IMPALA_KEY_STMT(tok, t_str) case Token:: tok:
#include "impala/tokenlist.h"
        case Token::ID:
        case Token::L_BRACE:
        case Token::SEMICOLON:
            return true;
        default:
            return false;
    }
}

const Type* Parser::try_type(const std::string& what) {
    const Type* type;
    if (is_type())
        type = parse_type();
    else {
        error("type", what);
        type = world.type_error();
        lex(); // eat away erroneous token
    }

    return type;
}

const Expr* Parser::try_expr(Prec prec, const std::string& what) {
    const Expr* expr;
    if (is_expr())
        expr = parse_expr(prec);
    else {
        error("expression", what);
        expr = new EmptyExpr(prev_loc);
        lex(); // eat away erroneous token
    }

    return expr;
}

const Stmt* Parser::try_stmt(const std::string& what) {
    const Stmt* stmt;
    if (is_stmt())
        stmt = parse_stmt();
    else {
        error("statement", what);
        stmt = new ExprStmt(new EmptyExpr(prev_loc), prev_loc.pos2());
        lex(); // eat away erroneous token
    }

    return stmt;
}

/*
 * expressions
 */

const Expr* Parser::parse_expr(Prec prec) {
    const Expr* lhs = la().is_prefix() ? parse_prefix_expr() : parse_primary_expr();

    while (true) {
        /*
         * (lhs  op  LA) op ...  on break  (current prec > lhs prec of LA)  -->  reduce
         *  lhs  op (LA  op ...) otherwise                                  -->  shift
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
    const Expr* rhs = try_expr(PrecTable::prefix_r[op], "prefix expression");

    return new PrefixExpr(op.pos1(), (PrefixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_infix_expr(const Expr* lhs) {
    Token op = lex();
    if (op == Token::QUESTION_MARK) {
        const Expr* t_expr = try_expr("infix expression");
        expect(Token::COLON, "conditional expression");
        const Expr* f_expr = try_expr(PrecTable::infix_r[op], "conditional expression");
        return new ConditionalExpr(lhs, t_expr, f_expr);
    }

    const Expr* rhs = try_expr(PrecTable::infix_r[op], "infix expression");

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
        call->set_pos2(prev_loc.pos2());

        return call;
    } else if (accept(Token::L_BRACKET)) {
        Position pos1 = prev_loc.pos1();
        const Expr* index = try_expr("indexing expression");
        expect(Token::R_BRACKET, "index expression");
        return new IndexExpr(pos1, lhs, index, prev_loc.pos2());
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
#include "impala/tokenlist.h"
        case Token::TRUE:
        case Token::FALSE:      return parse_literal();
        case Token::ID: {
            assert(!generic_lookup(la().symbol()));
            return new Id(lex());
        }
        case Token::L_PAREN: {
            eat(Token::L_PAREN);
            const Expr* expr = try_expr("primary expression");
            expect(Token::R_PAREN, "primary expression");
            return expr;
        }
        case Token::HASH:       return parse_tuple();
        case Token::LAMBDA:     return parse_fun_expr();
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
    Tuple* tuple = new Tuple(eat(Token::HASH).pos1());
    expect(Token::L_PAREN, "tuple");

    PARSE_COMMA_LIST
    (
        tuple->ops_.push_back(try_expr("tuple element")),
        Token::R_PAREN,
        "closing parenthesis of tuple"
    )

    tuple->set_pos2(prev_loc.pos2());

    return tuple;
}

const Expr* Parser::parse_fun_expr() {
    Position pos1 = eat(Token::LAMBDA).pos1();
    FunExpr* fun_expr = new FunExpr();
    parse_fun(fun_expr);

    return fun_expr;
}

} // namespace impala
