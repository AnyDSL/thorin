#include <algorithm>
#include <functional>
#include <sstream>
#include <iostream>

#include "anydsl2/util/array.h"
#include "anydsl2/util/assert.h"
#include "anydsl2/util/push.h"

#include "impala/ast.h"
#include "impala/lexer.h"
#include "impala/prec.h"
#include "impala/type.h"

using namespace anydsl2;

namespace impala {

class Parser {
public:
    Parser(TypeTable& typetable, std::istream& stream, const std::string& filename)
        : typetable(typetable)
        , lexer(stream, filename)
        , cur_loop(nullptr)
        , cur_fun(nullptr)
        , cur_var_handle(2) // reserve 0 for conditionals, 1 for mem
        , no_bars_(false)
        , generic_counter(0)
        , counter(0)
        , result_(true)
    {
        lookahead[0] = lexer.lex();
        lookahead[1] = lexer.lex();
    }

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
    const Expr* try_expr(Prec prec, const std::string& what, bool no_bars = false);
    const Expr* try_expr(const std::string& what) { return try_expr(BOTTOM, what, false); }
    const Stmt* try_stmt(const std::string& what);
    bool is_type(size_t lookahead = 0);
    bool is_expr();
    bool is_stmt();
    bool parse_prg(Scope*);
    const Scope* parse_scope();
    const Scope* try_scope(const std::string& context);
    const Scope* parse_stmt_as_scope(const std::string& what);
    void parse_generics_list(TypeDecls&);
    const Type* parse_type();
    const Type* parse_compound_type();
    const Type* parse_return_type();
    const VarDecl* parse_var_decl(bool is_param);
    const Proto* parse_proto();
    void parse_fun(Fun* fun);
    const Decl* parse_decl();

    void parse_comma_list(TokenKind delimiter, const char* context, std::function<void()> f) {
        if (la() != delimiter) {
            do { f(); }
            while ( accept(Token::COMMA) );
        }
        expect(delimiter, context);
    }

    // expressions
    bool is_infix();
    const Expr* parse_expr(Prec prec);
    const Expr* parse_expr() { return parse_expr(BOTTOM, false); }
    const Expr* parse_expr(Prec prec, bool no_bars) {
        ANYDSL2_PUSH(no_bars_, no_bars);
        return parse_expr(prec); 
    }
    const Expr* parse_prefix_expr();
    const Expr* parse_infix_expr(const Expr* lhs);
    const Expr* parse_postfix_expr(const Expr* lhs);
    const Expr* parse_primary_expr();
    const Expr* parse_literal();
    const Expr* parse_tuple();
    const FunExpr* parse_fun_expr();

    // statements
    const Stmt* parse_stmt();
    const ExprStmt* parse_expr_stmt();
    const Stmt* parse_decl_stmt_or_init_stmt();
    const Stmt* parse_if_else();
    const Stmt* parse_while();
    const Stmt* parse_do_while();
    const Stmt* parse_for();
    const Stmt* parse_foreach();
    const Stmt* parse_break();
    const Stmt* parse_continue();
    const Stmt* parse_return();
    const Stmt* parse_fun_stmt();
    const ScopeStmt* parse_scope_stmt();

    /// helper for condition in if/while/do-while
    const Expr* parse_cond(const std::string& what);

private:
    /// Consume next Token in input stream, fill look-ahead buffer, return consumed Token.
    Token lex();

    TypeTable& typetable;
    Lexer lexer;       ///< invoked in order to get next token
    Token lookahead[2];///< LL(2) look ahead
    const Loop* cur_loop;
    const Fun* cur_fun;
    size_t cur_var_handle;
    bool no_bars_;
    size_t generic_counter;
    int counter;
    anydsl2::Location prev_loc;
    bool result_;
};

//------------------------------------------------------------------------------

bool parse(TypeTable& typetable, std::istream& i, const std::string& filename, Scope* prg) {
    return Parser(typetable, i, filename).parse_prg(prg);
}

//------------------------------------------------------------------------------

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

bool Parser::is_type(size_t lookahead) {
    switch (la(lookahead)) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_##itype:
#include "impala/tokenlist.h"
        case Token::TYPE_int:
        case Token::TYPE_uint:
        case Token::TYPE_void:
        case Token::TYPE_noret:
        case Token::FN:
        case Token::SIGMA:
        case Token::ID:
            return true;
        default:
            return false;
    }
}

bool Parser::is_expr() {
    //identifier without a succeeding colon which is not a generic
    if (la() == Token::ID && la2() != Token::COLON)
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
        case Token::EXTERN:
        case Token::FN:
        case Token::ID:
        case Token::L_BRACE:
        case Token::SEMICOLON:
            return true;
        default:
            return false;
    }
}

/*
 * scope
 */

bool Parser::parse_prg(Scope* scope) {
    if (scope->empty())
        scope->set_pos1(la().pos1());

    while (true) {
        switch (la()) {
            case Token::SEMICOLON:      lex(); continue; // ignore semicolon
            case Token::END_OF_FILE:    scope->set_pos2(prev_loc.pos2()); return result();
            case Token::FN:             scope->stmts_.push_back(parse_fun_stmt()); continue;
            case Token::EXTERN:         scope->stmts_.push_back(parse_decl_stmt_or_init_stmt()); continue;
            default:                    lex(); continue; // consume token nobody wants
        }
    }
}

const Scope* Parser::try_scope(const std::string& context) {
    if (la() == Token::L_BRACE)
        return parse_scope();
    error("scope", context);
    return new Scope(prev_loc);
}

const Scope* Parser::parse_stmt_as_scope(const std::string& what) {
    if (la() == Token::L_BRACE)
        return parse_scope();
    auto scope = new Scope();
    scope->set_pos1(la().pos1());
    scope->stmts_.push_back(try_stmt(what));
    scope->set_pos2(prev_loc.pos2());
    return scope;
}

const Scope* Parser::parse_scope() {
    Scope* scope = new Scope();
    scope->set_pos1(eat(Token::L_BRACE).pos1());

    while (true) {
        if (accept(Token::SEMICOLON)) {
            continue; // ignore semicolon
        } else if (la() == Token::R_BRACE || la() == Token::END_OF_FILE) {
            break;
        } else if (is_stmt()) {
            scope->stmts_.push_back(parse_stmt());
        } else {
            lex(); // consume token nobody wants
        }
    }

    expect(Token::R_BRACE, "scope statement");
    scope->set_pos2(prev_loc.pos2());

    return scope;
}

/*
 * types
 */

const Type* Parser::try_type(const std::string& what) {
    const Type* type;
    if (is_type())
        type = parse_type();
    else {
        error("type", what);
        type = typetable.type_error();
        lex(); // eat away erroneous token
    }

    return type;
}

const Type* Parser::parse_type() {
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token::TYPE_##itype: lex(); return typetable.type_##itype();
#include "impala/tokenlist.h"

        case Token::TYPE_int:   lex(); return typetable.type_int32();
        case Token::TYPE_void:  lex(); return typetable.type_void();
        case Token::TYPE_noret: lex(); return typetable.noret();
        case Token::FN:
        case Token::SIGMA:             return parse_compound_type();
        // TODO generic refs
        case Token::ID: 
            return typetable.idtype(lex().symbol());
        default: ANYDSL2_UNREACHABLE;
    }
}

const Type* Parser::parse_compound_type() {
    bool fn = lex().kind() == Token::FN;

    std::vector<const Type*> elems;
    const char* error_str = fn ? "element types of fn" : "element types of sigma";
    expect(Token::L_PAREN, error_str);
    parse_comma_list(Token::R_PAREN, fn ?  "closing parenthesis of fn type" : "closing parenthesis of sigma type", [&] {
        elems.push_back(try_type(error_str));
    });

    if (fn && accept(Token::ARROW))
        elems.push_back(parse_return_type());

    if (fn) 
        return typetable.fntype(elems);
    return typetable.tupletype(elems);
}

const Type* Parser::parse_return_type() {
    const Type* ret_type = try_type("return type");
    if (ret_type->is_void())
        return typetable.fntype0();
    else
        return typetable.fntype1(ret_type);
}

/*
 * delcs
 */

const Decl* Parser::parse_decl() {
    switch (la()) {
        case Token::EXTERN: return parse_proto();
        default:            return parse_var_decl(false);
    }
}

const VarDecl* Parser::parse_var_decl(bool is_param) {
    Token tok = la();
    expect(Token::ID, "declaration");
    expect(Token::COLON, "declaration");
    const Type* type = try_type("declaration");

    return new VarDecl(cur_var_handle++, is_param, tok, type, prev_loc.pos2());
}

const Proto* Parser::parse_proto() {
    Position pos1 = la().pos1();
    eat(Token::EXTERN);
    Proto* proto = new Proto(try_id("prototype").symbol());
    std::vector<const Type*> types;

    expect(Token::L_PAREN, "prototype");
    parse_comma_list(Token::R_PAREN, "type list of prototype", [&] {
        types.push_back(try_type("type list of prototype function"));
    });
    expect(Token::ARROW, "prototype");

    const Type* ret_type = try_type("return type of prototype");
    if (ret_type->is_void())
        types.push_back(typetable.fntype0());
    else
        types.push_back(typetable.fntype1(ret_type));

    proto->orig_type_ = typetable.fntype(types);
    proto->set_loc(pos1, prev_loc.pos2());

    return proto;
}

void Parser::parse_generics_list(TypeDecls& typedecls) {
    if (accept(Token::LT))
        parse_comma_list(Token::GT, "generics list", [&] {
            typedecls.push_back(new TypeDecl(try_id("generic identifier")));
        });
}

void Parser::parse_fun(Fun* fun) {
    ANYDSL2_PUSH(cur_fun, fun);
    ANYDSL2_PUSH(cur_var_handle, cur_var_handle);

    fun->set_pos1(la().pos1());
    std::vector<const Type*> arg_types;
    expect(Token::L_PAREN, "function head");
    parse_comma_list(Token::R_PAREN, "parameter list", [&] {
        const VarDecl* param = parse_var_decl(true);
        fun->params_.push_back(param);
        arg_types.push_back(param->orig_type());
    });

    // return-continuation
    if (accept(Token::ARROW)) {
        Position pos1 = prev_loc.pos1();
        arg_types.push_back(parse_return_type());
        Position pos2 = prev_loc.pos2();
        fun->params_.push_back(new VarDecl(cur_var_handle++, true, Token(pos1, "return"), arg_types.back(), pos2));
    }

    fun->orig_type_ = typetable.fntype(arg_types);
    fun->body_ = try_scope("body of function");
    fun->set_pos2(prev_loc.pos2());
}

/*
 * expressions
 */

const Expr* Parser::try_expr(Prec prec, const std::string& what, bool no_bars) {
    const Expr* expr;
    if (is_expr())
        expr = parse_expr(prec, no_bars);
    else {
        error("expression", what);
        expr = new EmptyExpr(prev_loc);
        lex(); // eat away erroneous token
    }

    return expr;
}

bool Parser::is_infix() {
    bool infix = la().is_infix();
    if (no_bars_ && infix)
        return !la() == Token::OR && !la() == Token::L_O;
    return infix;
}

const Expr* Parser::parse_expr(Prec prec) {
    const Expr* lhs = la().is_prefix() ? parse_prefix_expr() : parse_primary_expr();

    while (true) {
        /*
         * (lhs  op  LA) op ...  on break  (current prec > lhs prec of LA)  -->  reduce
         *  lhs  op (LA  op ...) otherwise                                  -->  shift
         */

        if (is_infix()) {
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
    if (la() == Token::OR || la() == Token::L_O)
        return parse_fun_expr();
    Token op = lex();
    const Expr* rhs = try_expr(PrecTable::prefix_r[op], "prefix expression");

    return new PrefixExpr(op.pos1(), (PrefixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_infix_expr(const Expr* lhs) {
    Token op = lex();
    if (op == Token::QUESTION_MARK) {
        const Expr* t_expr = la() == Token::ID ? parse_expr() : try_expr("infix expression");
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
        parse_comma_list(Token::R_PAREN, "arguments of a function call", [&] {
            call->append_arg(try_expr("argument of function call"));
        });
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
        case Token::ID:         return new Id(lex());
        case Token::L_PAREN: {
            eat(Token::L_PAREN);
            const Expr* expr = try_expr("primary expression");
            expect(Token::R_PAREN, "primary expression");
            return expr;
        }
        case Token::HASH:       return parse_tuple();
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
    parse_comma_list(Token::R_PAREN, "closing parenthesis of tuple", [&] {
        tuple->ops_.push_back(try_expr("tuple element"));
    });

    tuple->set_pos2(prev_loc.pos2());

    return tuple;
}

const FunExpr* Parser::parse_fun_expr() {
    FunExpr* e = new FunExpr(typetable);
    e->set_pos1(la().pos1());
    Fun* fun = e->fun_;

    ANYDSL2_PUSH(cur_fun, fun);
    ANYDSL2_PUSH(cur_var_handle, cur_var_handle);

    fun->set_pos1(la().pos1());
    std::vector<const Type*> arg_types;
    if (accept(Token::OR)) {
        parse_comma_list(Token::OR, "parameter list of function expression", [&] {
            const VarDecl* param = parse_var_decl(true);
            fun->params_.push_back(param);
            arg_types.push_back(param->orig_type());
        });
    } else
        expect(Token::L_O, "parameter list of function expression");

    // return-continuation
    if (accept(Token::ARROW)) {
        Position pos1 = prev_loc.pos1();
        arg_types.push_back(parse_return_type());
        Position pos2 = prev_loc.pos2();
        fun->params_.push_back(new VarDecl(cur_var_handle++, true, Token(pos1, "return"), arg_types.back(), pos2));
    }

    fun->orig_type_ = typetable.fntype(arg_types);
    fun->body_ = try_scope("body of function");
    fun->set_pos2(prev_loc.pos2());

    e->fun_->extern_ = false;
    e->fun_->symbol_ = "<lambda>";
    e->set_pos2(prev_loc.pos2());

    return e;
}

/*
 * statements
 */

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

const Stmt* Parser::parse_stmt() {
    if (is_expr())
        return parse_expr_stmt();

    if (la() == Token::EXTERN 
            || (la() == Token::ID && la2() == Token::COLON))
        return parse_decl_stmt_or_init_stmt();

    switch (la()) {
        case Token::BREAK:     return parse_break();
        case Token::CONTINUE:  return parse_continue();
        case Token::FN:        return parse_fun_stmt();
        case Token::DO:        return parse_do_while();
        case Token::FOR:       return parse_for();
        case Token::FOREACH:   return parse_foreach();
        case Token::IF:        return parse_if_else();
        case Token::L_BRACE:   return parse_scope_stmt();
        case Token::RETURN:    return parse_return();
        case Token::WHILE:     return parse_while();
        case Token::SEMICOLON: return new ScopeStmt(lex().loc());
        case Token::ELSE:
            error("'else' without matching 'if'", "statement");
            return new ScopeStmt(lex().loc());
        default:               ANYDSL2_UNREACHABLE;
    }
}

const ScopeStmt* Parser::parse_scope_stmt() {
    ScopeStmt* s = new ScopeStmt();
    s->set_pos1(la().pos1());
    s->scope_ = parse_scope();
    s->set_pos2(prev_loc.pos2());
    return s;
}

const ExprStmt* Parser::parse_expr_stmt() {
    const Expr* expr = parse_expr(); // discard val
    expect(Token::SEMICOLON, "the end of an expression statement");

    return new ExprStmt(expr, prev_loc.pos2());
}

const Stmt* Parser::parse_decl_stmt_or_init_stmt() {
    const Decl* decl = parse_decl();
    if (const auto var_decl = decl->isa<VarDecl>()) {
        if (accept(Token::ASGN)) {
            auto init = new InitStmt(var_decl, try_expr("right-hand side of an initialization"));
            expect(Token::SEMICOLON, "the end of an initialization statement");
            return init;
        }
    }

    auto decl_stmt = new DeclStmt(decl);
    expect(Token::SEMICOLON, "the end of an declaration statement");
    return decl_stmt;
}

const Expr* Parser::parse_cond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    const Expr* cond = try_expr("condition in " + what);
    expect(Token::R_PAREN, "condition in " + what);
    return cond;
}

const Stmt* Parser::parse_if_else() {
    auto ifelse = new IfElseStmt();
    ifelse->set_pos1(eat(Token::IF).pos1());
    ifelse->cond_ = parse_cond("if statement");
    ifelse->then_scope_ = parse_stmt_as_scope("if clause");
    ifelse->else_scope_ = accept(Token::ELSE) ? parse_stmt_as_scope("else clause") : new Scope(prev_loc);
    ifelse->set_pos2(prev_loc.pos2());
    return ifelse;
}

const Stmt* Parser::parse_while() {
    auto loop = new ForStmt();
    loop->set_pos1(eat(Token::WHILE).pos1());
    loop->init_ = new ExprStmt(new EmptyExpr(loop->pos1()), loop->pos1());
    loop->cond_ = parse_cond("while statement");
    loop->step_ = new EmptyExpr(loop->pos1());
    ANYDSL2_PUSH(cur_loop, loop);
    loop->body_ = parse_stmt_as_scope("body of while statement");
    loop->set_pos2(prev_loc.pos2());
    return loop;
}

const Stmt* Parser::parse_do_while() {
    DoWhileStmt* loop = new DoWhileStmt();
    loop->set_pos1(eat(Token::DO).pos1());
    ANYDSL2_PUSH(cur_loop, loop);
    loop->body_ = parse_stmt_as_scope("body of do-while statement");
    expect(Token::WHILE, "do-while statement");
    loop->cond_ = parse_cond("do-while statement");
    expect(Token::SEMICOLON, "do-while statement");
    loop->set_pos2(prev_loc.pos2());

    return loop;
}

const Stmt* Parser::parse_for() {
    ForStmt*  loop = new ForStmt();
    loop->set_pos1(eat(Token::FOR).pos1());
    expect(Token::L_PAREN, "for statement");

    ANYDSL2_PUSH(cur_loop, loop);

    // clause 1: decl or expr_opt ';'
    if (la2() == Token::COLON)
        loop->init_ = parse_decl_stmt_or_init_stmt();
    else if (is_expr())
        loop->init_ = parse_expr_stmt();
    else  {
        if (!accept(Token::SEMICOLON))
            error("expression or delcaration statement", "first clause in for statement");
        loop->init_ = new ExprStmt(new EmptyExpr(loop->pos1()), loop->pos1());
    }

    // clause 2: expr_opt ';'
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed, but create true cond
        loop->cond_ = new Literal(prev_loc, Literal::LIT_bool, Box(true));
    } else if (is_expr()) {
        loop->cond_ = parse_expr();
        expect(Token::SEMICOLON, "second clause in for statement");
    } else {
        error("expression or nothing", 
                "second clause in for statement");
        loop->cond_ = new Literal(prev_loc, Literal::LIT_bool, Box(true));
    }

    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
        loop->step_ = new EmptyExpr(prev_loc);
    } else if (is_expr()) {
        loop->step_ = parse_expr();
        expect(Token::R_PAREN, "for statement");
    } else {
        error("expression or nothing",
                "third clause in for statement");
        loop->step_ = new EmptyExpr(prev_loc);
    }

    loop->body_ = parse_stmt_as_scope("body of for statement");
    loop->set_pos2(prev_loc.pos2());

    return loop;
}

const Stmt* Parser::parse_foreach() {
    ForeachStmt* foreach = new ForeachStmt();
    foreach->set_pos1(eat(Token::FOREACH).pos1());

    const Expr* expr = try_expr(BOTTOM, "generator call in for-each statement", true);
    if (const Call* call = expr->isa<Call>()) {
        foreach->call_ = call;
    } else {
        error("generator call", " for-each statement");
        delete expr;
        foreach->call_ = nullptr; // TODO
    }

    foreach->fun_expr_ = parse_fun_expr();
    foreach->set_pos2(prev_loc.pos2());

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
        expr = nullptr;
        eat(Token::SEMICOLON);
    }

    return new ReturnStmt(pos1, expr, cur_fun, prev_loc.pos2());
}

const Stmt* Parser::parse_fun_stmt() {
    FunStmt* s = new FunStmt(typetable);
    s->set_pos1(eat(Token::FN).pos1());
    s->fun_->extern_ = accept(Token::EXTERN);
    s->fun_->symbol_ = try_id("function identifier").symbol();
    parse_generics_list(s->fun_->generics_);
    parse_fun(s->fun_);
    s->set_pos2(prev_loc.pos2());

    return s;
}

} // namespace impala
