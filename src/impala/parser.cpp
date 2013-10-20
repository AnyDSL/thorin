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

#define TYPE \
         Token::TYPE_int8: \
    case Token::TYPE_int16: \
    case Token::TYPE_int32: \
    case Token::TYPE_int64: \
    case Token::TYPE_float: \
    case Token::TYPE_double: \
    case Token::TYPE_bool: \
    case Token::FN: \
    case Token::L_PAREN: \
    case Token::ID

#define ITEM \
         Token::FN: \
    case Token::EXTERN: \
    case Token::CLASS: \
    case Token::STRUCT

#define DECL \
         Token::VAL: \
    case Token::VAR

#define EXPR \
         Token::LIT_int8: \
    case Token::LIT_int16: \
    case Token::LIT_int32: \
    case Token::LIT_int64: \
    case Token::LIT_float: \
    case Token::LIT_double: \
    case Token::TRUE: \
    case Token::FALSE: \
    case Token::ADD: \
    case Token::SUB: \
    case Token::MUL: \
    case Token::AND: \
    case Token::NOT: \
    case Token::L_N: \
    case Token::INC: \
    case Token::DEC: \
    case Token:: OR: \
    case Token::L_O: \
    case Token::ID: \
    case Token::RUN: \
    case Token::L_PAREN
    
#define STMT_NO_EXPR \
         DECL: \
    case ITEM: \
    case Token::IF: \
    case Token::FOR: \
    case Token::FOREACH: \
    case Token::DO: \
    case Token::WHILE: \
    case Token::BREAK: \
    case Token::CONTINUE: \
    case Token::RETURN: \
    case Token::L_BRACE

#define STMT \
        STMT_NO_EXPR: \
    case EXPR

using namespace anydsl2;

namespace impala {

class Parser;

template<class T>
class Loc {
public:
    inline Loc(Parser& parser, T* node);
    inline ~Loc();

    operator T*() { return node_; }
    T* operator -> () { return node_; }

private:
    Parser& parser_;
    T* node_;
};

class Parser {
public:
    Parser(TypeTable& typetable, std::istream& stream, const std::string& filename)
        : typetable(typetable)
        , lexer(stream, filename)
        , cur_loop(nullptr)
        , cur_fun(nullptr)
        , cur_var_handle(2) // reserve 0 for conditionals, 1 for mem
        , no_bars_(false)
        , result_(true)
    {
        lookahead[0] = lexer.lex();
        lookahead[1] = lexer.lex();
    }

    const Token& la(size_t i) const { return lookahead[i]; }
    const Token& la () const { return lookahead[0]; }
    const Token& la2() const { return lookahead[1]; }
    anydsl2::Location prev_loc() const { return prev_loc_; }

#ifdef NDEBUG
    Token eat(TokenKind /*what*/) { return lex(); }
#else
    Token eat(TokenKind what) { assert(what == la() && "internal parser error"); return lex(); }
#endif

    bool accept(TokenKind tok);
    bool expect(TokenKind tok, const std::string& context);
    void error(const std::string& what, const std::string& context);
    bool result() const { return result_; }
    template<class T>
    Loc<T> loc(T* node) { return Loc<T>(*this, node); }
    const ExprStmt* lex_as_empty_expr_stmt() { return new ExprStmt(new EmptyExpr(lex().loc())); }
    const ExprStmt* create_empty_expr_stmt() { return new ExprStmt(new EmptyExpr(prev_loc())); }

    // misc
    Token try_id(const std::string& what);
    bool is_expr() const {
        switch (la()) {
            case EXPR: return true;
            default:   return false;
        }
    }
    bool parse_prg(Scope*);
    const Scope* parse_scope();
    const Scope* try_scope(const std::string& context);
    const Scope* parse_stmt_as_scope(const std::string& what);
    void parse_generics_list(GenericDecls&);
    const Type* parse_type();
    const Type* try_type_postfix(const Type* type);
    const Type* parse_fn_type();
    const Type* parse_tuple_type();
    const Type* parse_return_type();
    const VarDecl* parse_var_decl(bool is_param);
    void parse_proto(Proto* proto);
    void parse_fun(Fun* fun);
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
    const FunExpr* parse_fun_expr();

    // statements
    const Stmt* parse_stmt();
    const ItemStmt* parse_item_stmt();
    const ExprStmt* parse_expr_stmt();
    const Stmt* parse_init_stmt();
    const Stmt* parse_if_else();
    const Stmt* parse_while();
    const Stmt* parse_do_while();
    const Stmt* parse_for();
    const Stmt* parse_foreach();
    const Stmt* parse_break();
    const Stmt* parse_continue();
    const Stmt* parse_return();
    const Stmt* parse_label_stmt();
    const ScopeStmt* parse_scope_stmt();

    // items
    const Item* parse_item();
    const ProtoItem* parse_proto_item();
    const FunItem* parse_fun_item();
    const TraitItem* parse_trait_item();

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
    anydsl2::Location prev_loc_;
    bool result_;
};

//------------------------------------------------------------------------------

template<class T>
Loc<T>::Loc(Parser& parser, T* node)
    : parser_(parser)
    , node_(node)
{
    node_->set_pos1(parser_.la().pos1());
}
template<class T>
Loc<T>::~Loc() { node_->set_pos2(parser_.prev_loc().pos2()); }

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
    prev_loc_ = result.loc();     // remember previous location

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

/*
 * scope
 */

bool Parser::parse_prg(Scope* scope) {
    if (scope->empty())
        scope->set_pos1(la().pos1());

    while (true) {
        switch (la()) {
            case Token::END_OF_FILE: scope->set_pos2(prev_loc().pos2()); return result();
            case ITEM:               scope->stmts_.push_back(parse_item_stmt()); continue;
            case Token::SEMICOLON:   lex(); continue;
            default:
                error("item", "program");
                lex();
                continue; // consume token nobody wants
        }
    }
}

const Scope* Parser::try_scope(const std::string& context) {
    if (la() == Token::L_BRACE)
        return parse_scope();
    error("scope", context);
    return new Scope(prev_loc());
}

const Scope* Parser::parse_stmt_as_scope(const std::string& what) {
    if (la() == Token::L_BRACE)
        return parse_scope();
    switch (la()) {
        case STMT: {
            auto scope = loc(new Scope());
            scope->stmts_.push_back(parse_stmt());
            return scope;
        }
        default: return new Scope(prev_loc());
    }
}

const Scope* Parser::parse_scope() {
    eat(Token::L_BRACE);
    auto scope = loc(new Scope());
    while (true) {
        switch (la()) {
            case Token::SEMICOLON:  lex(); continue; // ignore semicolon
            case STMT:              scope->stmts_.push_back(parse_stmt()); continue;
            //case STMT_NO_EXPR:      scope->stmts_.push_back(parse_stmt()); continue;
            //case EXPR: {
                //auto expr = parse_expr();
                //if (accept(Token::SEMICOLON)) {
                    //scope->stmts_.push_back(new ExprStmt(expr));
                    //scope->set_loc(expr->pos1(), prev_loc().pos2());
                    //continue;
                //} else {
                    //assert(false && "TODO");
                    ////scope->expr_ = expr;
                //}
                //// FALLTHROUGH
            //} 
            default:
                expect(Token::R_BRACE, "scope statement");
                return scope;
        }
    }
}

/*
 * types
 */

const Type* Parser::parse_type() {
    const Type* type;
    switch (la()) {
#define IMPALA_TYPE(itype, atype) \
        case Token::TYPE_##itype:   lex(); type = typetable.type_##itype(); break;
#include "impala/tokenlist.h"
        case Token::TYPE_int:       lex(); type = typetable.type_int32(); break;
        case Token::TYPE_void:      lex(); type = typetable.type_void(); break;
        case Token::TYPE_noret:     lex(); type = typetable.noret(); break;
        case Token::FN:                    type = parse_fn_type(); break;
        case Token::L_PAREN:               type = parse_tuple_type(); break;
        case Token::ID:                    type = typetable.idtype(lex().symbol()); break;
        default:                    error("type", ""); lex(); return typetable.type_error();
    }
    return try_type_postfix(type);
}

const Type* Parser::try_type_postfix(const Type* type) {
    if (accept(Token::L_BRACKET)) {
        // parsing an array
        size_t dim = 1;
        while (accept(Token::COMMA))
            ++dim;
        expect(TokenKind::R_BRACKET, "end of array declaration");
        return typetable.arraytype(dim, type);
    }
    return type;
}

const Type* Parser::parse_fn_type() {
    eat(Token::FN);
    std::vector<const Type*> elems;
    expect(Token::L_PAREN, "parameter list of function type");
    parse_comma_list(Token::R_PAREN, "closing parenthesis of function type", [&]{ elems.push_back(parse_type()); });

    if (accept(Token::ARROW))
        elems.push_back(parse_return_type());

    return typetable.fntype(elems);
}

const Type* Parser::parse_tuple_type() {
    eat(Token::L_PAREN);
    std::vector<const Type*> elems;
    parse_comma_list(Token::R_PAREN, "closing parenthesis of sigma type", [&]{ elems.push_back(parse_type()); });

    return typetable.tupletype(elems);
}

const Type* Parser::parse_return_type() {
    auto ret_type = parse_type();
    return ret_type->is_void() ? typetable.fntype({}) : typetable.fntype({ret_type});
}

/*
 * delcs
 */

const VarDecl* Parser::parse_var_decl(bool is_param) {
    Token tok = la();
    expect(Token::ID, "declaration");
    const Type* type;
    if (!is_param) {
        type = accept(Token::COLON) ? parse_type() : nullptr;
    } else {
        expect(Token::COLON, "declaration");
        type = parse_type();
    }
    return new VarDecl(cur_var_handle++, is_param, tok, type, prev_loc().pos2());
}

void Parser::parse_proto(Proto* proto) {
    std::vector<const Type*> types;
    loc(proto);

    expect(Token::L_PAREN, "prototype");
    parse_comma_list(Token::R_PAREN, "type list", [&] {
        types.push_back(parse_type());
    });
    expect(Token::ARROW, "prototype");

    const Type* ret_type = parse_type();
    if (ret_type->is_void())
        types.push_back(typetable.fntype({}));
    else
        types.push_back(typetable.fntype({ret_type}));

    proto->orig_type_ = typetable.fntype(types);
}

void Parser::parse_generics_list(GenericDecls& generic_decls) {
    if (accept(Token::LT))
        parse_comma_list(Token::GT, "generics list", [&] {
            generic_decls.push_back(new GenericDecl(try_id("generic identifier")));
        });
}

void Parser::parse_fun(Fun* f) {
    ANYDSL2_PUSH(cur_fun, f);
    ANYDSL2_PUSH(cur_var_handle, cur_var_handle);
    auto fun = loc(f);
    std::vector<const Type*> arg_types;
    expect(Token::L_PAREN, "function head");
    parse_comma_list(Token::R_PAREN, "parameter list", [&] {
        const VarDecl* param = parse_var_decl(true);
        fun->params_.push_back(param);
        arg_types.push_back(param->orig_type());
    });

    // return-continuation
    if (accept(Token::ARROW)) {
        Position pos1 = prev_loc().pos1();
        arg_types.push_back(parse_return_type());
        Position pos2 = prev_loc().pos2();
        fun->params_.push_back(new VarDecl(cur_var_handle++, true, Token(pos1, "return"), arg_types.back(), pos2));
    }

    fun->orig_type_ = typetable.fntype(arg_types);
    fun->body_ = try_scope("body of function");
}

/*
 * expressions
 */

bool Parser::is_infix() {
    bool infix = la().is_infix();
    if (no_bars_ && infix)
        return la() != Token::OR && la() != Token::L_O;
    return infix;
}

const Expr* Parser::parse_expr(Prec prec) {
    auto lhs = la().is_prefix() ? parse_prefix_expr() : parse_primary_expr();

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
    auto rhs = parse_expr(PrecTable::prefix_r[op]);

    return new PrefixExpr(op.pos1(), (PrefixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_infix_expr(const Expr* lhs) {
    Token op = lex();
    if (op == Token::QUESTION_MARK) {
        auto t_expr = parse_expr();
        expect(Token::COLON, "conditional expression");
        auto f_expr = parse_expr(PrecTable::infix_r[op]);
        return new ConditionalExpr(lhs, t_expr, f_expr);
    }

    auto rhs = parse_expr(PrecTable::infix_r[op]);

    return new InfixExpr(lhs, (InfixExpr::Kind) op.kind(), rhs);
}

const Expr* Parser::parse_postfix_expr(const Expr* lhs) {
    if (accept(Token::L_PAREN)) {
        auto call = loc(new Call());
        call->ops_.push_back(lhs);
        parse_comma_list(Token::R_PAREN, "arguments of a function call", [&]{ call->append_arg(parse_expr()); });
        return call;
    } else if (accept(Token::L_BRACKET)) {
        auto expr = loc(new IndexExpr());
        expr->ops_.push_back(lhs);
        expr->ops_.push_back(parse_expr());
        expect(Token::R_BRACKET, "index expression");
        return expr;
    } else {
        auto expr = loc(new PostfixExpr());
        expr->ops_.push_back(lhs);
        assert(la() == Token::INC || la() == Token::DEC); 
        expr->kind_ = (PostfixExpr::Kind) lex().kind();
        return expr;
    }
}

const Expr* Parser::parse_primary_expr() {
    switch (la() ) {
        case Token::L_PAREN: {
            Position pos1 = lex().pos1();
            auto expr = parse_expr();
            if (accept(Token::COMMA)) {
                auto tuple = new Tuple();
                tuple->set_pos1(pos1);
                tuple->ops_.push_back(expr);
                parse_comma_list(Token::R_PAREN, "elements of tuple expression", [&]{ tuple->ops_.push_back(parse_expr()); });
                tuple->set_pos2(prev_loc().pos2());
                return tuple;
            } else {
                expect(Token::R_PAREN, "primary expression");
                return expr;
            }
        }
#define IMPALA_LIT(itype, atype) \
        case Token::LIT_##itype:
#include "impala/tokenlist.h"
        case Token::TRUE:
        case Token::FALSE:      return parse_literal();
        case Token::ID:         return new Id(lex());
        default:                error("expression", ""); return new EmptyExpr(lex().loc());
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

const FunExpr* Parser::parse_fun_expr() {
    auto e = loc(new FunExpr(typetable));
    auto fun = loc(e->fun_.get());

    ANYDSL2_PUSH(cur_fun, fun);
    ANYDSL2_PUSH(cur_var_handle, cur_var_handle);

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
        Position pos1 = prev_loc().pos1();
        arg_types.push_back(parse_return_type());
        Position pos2 = prev_loc().pos2();
        fun->params_.push_back(new VarDecl(cur_var_handle++, true, Token(pos1, "return"), arg_types.back(), pos2));
    }

    fun->orig_type_ = typetable.fntype(arg_types);
    fun->body_ = try_scope("body of function");
    e->fun_->extern_ = false;
    e->fun_->symbol_ = "<lambda>";

    return e;
}

/*
 * statements
 */

const Stmt* Parser::parse_stmt() {
    if (la() == Token::ID && la2() == Token::COLON)
        return parse_label_stmt();

    switch (la()) {
        case DECL:              return parse_init_stmt();
        case EXPR:              return parse_expr_stmt();
        case ITEM:              return parse_item_stmt();
        case Token::BREAK:      return parse_break();
        case Token::CONTINUE:   return parse_continue();
        case Token::DO:         return parse_do_while();
        case Token::FOR:        return parse_for();
        case Token::FOREACH:    return parse_foreach();
        case Token::IF:         return parse_if_else();
        case Token::L_BRACE:    return parse_scope_stmt();
        case Token::RETURN:     return parse_return();
        case Token::WHILE:      return parse_while();
        case Token::SEMICOLON:                                                      return lex_as_empty_expr_stmt();
        case Token::ELSE:       error("'else' without matching 'if'", "statement"); return lex_as_empty_expr_stmt();
        default:                error("statement", "");                             return lex_as_empty_expr_stmt();
    }
}

const ScopeStmt* Parser::parse_scope_stmt() {
    auto s = loc(new ScopeStmt());
    s->scope_ = parse_scope();
    return s;
}

const ItemStmt* Parser::parse_item_stmt() {
    auto s = loc(new ItemStmt());
    s->item_ = parse_item();
    return s;
}

const ExprStmt* Parser::parse_expr_stmt() {
    auto s = loc(new ExprStmt());
    s->expr_ = parse_expr();
    expect(Token::SEMICOLON, "the end of an expression statement");
    return s;
}

const Stmt* Parser::parse_init_stmt() {
    lex(); // eat val/var
    auto s = loc(new InitStmt());
    s->var_decl_ = parse_var_decl(false);
    if (accept(Token::ASGN))
        s->init_ = parse_expr();

    expect(Token::SEMICOLON, "the end of an initialization statement");
    return s;
}

const Expr* Parser::parse_cond(const std::string& what) {
    expect(Token::L_PAREN, "condition in " + what);
    auto cond = parse_expr();
    expect(Token::R_PAREN, "condition in " + what);
    return cond;
}

const Stmt* Parser::parse_if_else() {
    eat(Token::IF);
    auto ifelse = loc(new IfElseStmt());
    ifelse->cond_ = parse_cond("if statement");
    ifelse->then_scope_ = parse_stmt_as_scope("if clause");
    ifelse->else_scope_ = accept(Token::ELSE) ? parse_stmt_as_scope("else clause") : new Scope(prev_loc());
    return ifelse;
}

const Stmt* Parser::parse_label_stmt() {
    Token tok = eat(Token::ID);
    eat(Token::COLON);
    Loop* loop = nullptr;
    switch (la()) {
        case Token::DO:     loop = (Loop*) parse_do_while(); break;
        case Token::FOR:    loop = (Loop*) parse_for();      break;
        case Token::WHILE:  loop = (Loop*) parse_while();    break;
        default:            error("for statement after label", ""); return create_empty_expr_stmt();

    }

    loop->set_pos1(tok.pos1());
    loop->label_ = tok.symbol();

    return loop;
}

const Stmt* Parser::parse_while() {
    eat(Token::WHILE);
    auto loop = loc(new ForStmt());
    loop->init_ = create_empty_expr_stmt();
    loop->cond_ = parse_cond("while statement");
    loop->step_ = new EmptyExpr(loop->pos1());
    ANYDSL2_PUSH(cur_loop, loop);
    loop->body_ = parse_stmt_as_scope("body of while statement");

    return loop;
}

const Stmt* Parser::parse_do_while() {
    eat(Token::DO);
    auto loop = loc(new DoWhileStmt());
    ANYDSL2_PUSH(cur_loop, loop);
    loop->body_ = parse_stmt_as_scope("body of do-while statement");
    expect(Token::WHILE, "do-while statement");
    loop->cond_ = parse_cond("do-while statement");
    expect(Token::SEMICOLON, "do-while statement");

    return loop;
}

const Stmt* Parser::parse_for() {
    eat(Token::FOR);
    auto loop = loc(new ForStmt());
    expect(Token::L_PAREN, "for statement");

    ANYDSL2_PUSH(cur_loop, loop);

    // clause 1: decl or expr_opt ';'
    switch (la()) {
        case DECL:              loop->init_ = parse_init_stmt(); break;
        case EXPR:              loop->init_ = parse_expr_stmt(); break;
        case Token::SEMICOLON:  loop->init_ = lex_as_empty_expr_stmt(); break;
        default: error("expression or delcaration statement", "first clause in for statement");
    }

    // clause 2: expr_opt ';'
    if (accept(Token::SEMICOLON)) { 
        // do nothing: no expr given, semicolon consumed, but create true cond
        loop->cond_ = new Literal(prev_loc(), Literal::LIT_bool, Box(true));
    } else if (is_expr()) {
        loop->cond_ = parse_expr();
        expect(Token::SEMICOLON, "second clause in for statement");
    } else {
        error("expression or nothing", 
                "second clause in for statement");
        loop->cond_ = new Literal(prev_loc(), Literal::LIT_bool, Box(true));
    }

    // clause 3: expr_opt ';'
    if (accept(Token::R_PAREN)) { 
        // do nothing: no expr given, semicolon consumed
        loop->step_ = new EmptyExpr(prev_loc());
    } else if (is_expr()) {
        loop->step_ = parse_expr();
        expect(Token::R_PAREN, "for statement");
    } else {
        error("expression or nothing",
                "third clause in for statement");
        loop->step_ = new EmptyExpr(prev_loc());
    }

    loop->body_ = parse_stmt_as_scope("body of for statement");

    return loop;
}

const Stmt* Parser::parse_foreach() {
    eat(Token::FOREACH);
    auto foreach = loc(new ForeachStmt());

    auto expr = parse_expr(BOTTOM, true);
    if (const Call* call = expr->isa<Call>()) {
        foreach->call_ = call;
    } else {
        error("generator call", " for-each statement");
        delete expr;
        foreach->call_ = nullptr; // TODO
    }

    foreach->fun_expr_ = parse_fun_expr();

    return foreach;
}

const Stmt* Parser::parse_break() {
    auto s = loc(new BreakStmt());
    s->loop_ = cur_loop;
    eat(Token::BREAK).pos1();
    expect(Token::SEMICOLON, "break statement");
    return s;
}

const Stmt* Parser::parse_continue() {
    auto s = loc(new ContinueStmt());
    s->loop_ = cur_loop;
    eat(Token::CONTINUE);
    expect(Token::SEMICOLON, "continue statement");
    return s;
}

const Stmt* Parser::parse_return() {
    auto ret = loc(new ReturnStmt());
    ret->fun_ = cur_fun;
    eat(Token::RETURN);
    if (accept(Token::SEMICOLON))
        ret->expr_ = nullptr;
    else {
        ret->expr_ = parse_expr();
        expect(Token::SEMICOLON, "return statement");
    }

    return ret;
}

/*
 * items
 */

const Item* Parser::parse_item() {
    switch (la()) {
        case Token::FN:     return parse_fun_item();
        case Token::EXTERN: return parse_proto_item();
        case Token::STRUCT:
        case Token::CLASS:  return parse_trait_item();
        default:            ANYDSL2_UNREACHABLE;
    }
}

const ProtoItem* Parser::parse_proto_item() {
    eat(Token::EXTERN);
    auto proto = loc(new ProtoItem(try_id("prototype").symbol()));
    parse_proto(proto->proto_ );
    expect(Token::SEMICOLON, "end of proto expected");
    return proto;
}

const FunItem* Parser::parse_fun_item() {
    auto fi = loc(new FunItem(typetable));
    fi->fun_->extern_ = accept(Token::EXTERN);
    eat(Token::FN);
    fi->fun_->symbol_ = try_id("function identifier").symbol();
    parse_generics_list(fi->fun_->generics_);
    parse_fun(fi->fun_);
    return fi;
}

const TraitItem* Parser::parse_trait_item() {
    auto i = loc(new TraitItem(typetable));
    // TODO
    expect(Token::CLASS, "trait");
    return i;
}

} // namespace impala

