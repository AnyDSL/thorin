#ifndef IMPALA_PARSER_H
#define IMPALA_PARSER_H

#include <boost/unordered_map.hpp>

#include "impala/ast.h"
#include "impala/lexer.h"

namespace anydsl {
    class CExpr;
    class Def;
    class Lambda;
    class Param;
    class World;
}

namespace impala {

class Loop;

enum Prec {
    BOTTOM,
    ASGN,
    COND,
    L_O,
    L_A,
    OR,
    XOR,
    AND,
    EQ,
    REL,
    SHIFT,
    ADD,
    MUL,
    UNARY,
    POSTFIX,
    TOP,
    NUM_PREC
};

struct BinPrec {
    Prec l;
    Prec r;

    BinPrec() {}
    BinPrec(Prec l, Prec r) : l(l), r(r) {}
};

typedef Prec Type2Prec[Token::NUM_TOKENS];
typedef BinPrec Type2BinPrec[Token::NUM_TOKENS];

class Parser {
public:

    /*
     * constructor
     */

    Parser(anydsl::World& world, std::istream& stream, const std::string& filename);

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

    // misc
    const Prg* parse();
    Token parseId();
    const Type* parseType();
    void parseParam();
    const Decl* parseDecl();
    void parseGlobals();
    void parseFct();

    // expressions
    bool isExpr();
    const Expr* tryExpr();
    const Expr* parseExpr(Prec prec);
    const Expr* parseExpr() { return parseExpr(BOTTOM); }
    const Expr* parsePrefixExpr();
    const Expr* parseInfixExpr(const Expr* left);
    const Expr* parsePostfixExpr(const Expr* left);
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

    /// Stmt+
    const Stmt* parseStmtList();
    /// '{' stmt '}'
    const Stmt* parseScopeBody();
    /// stmt which is \em not a compound-stmt \em or '{' stmt '}' \em without pushing/poping of scopes
    const Stmt* parseScope();
    /// (push scope) '{' stmt '}' (pop scope)
    const Stmt* parseCompoundStmt();

    /// helper for condition in if/while/do-while
    const Expr* parseCond(const std::string& what);

private:

    int nextId();

    /// Consume next Token in input stream, fill look-ahead buffer, return consumed Token.
    Token lex();

    /*
     * data
     */

    Lexer lexer_;       ///< invoked in order to get next token
    Token lookahead_[2];///< LL(2) look ahead
    const Loop* loop_;
    int counter_;
    anydsl::Location prevLoc_;

    static Type2Prec    preRPrec_; ///< right precedence -- for unary prefix operators
    static Type2BinPrec binPrec_;  ///< left and right precedences -- for binary operators
    static Type2Prec    postLPrec_;///< left precedence -- for unary postfix operators
};

} // namespace impala

#endif
