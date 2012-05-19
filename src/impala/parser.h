#ifndef IMPALA_PARSER_H
#define IMPALA_PARSER_H

#include "impala/emitter.h"
#include "impala/lexer.h"

namespace anydsl {
    class CExpr;
    class Def;
    class Lambda;
    class Param;
    class World;
}

namespace impala {

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
    anydsl::Lambda* parse();
    Token parseId();
    anydsl::Type* parseType();
    void parseParam();
    Value parseDecl();
    void parseGlobals();
    void parseFct();
    void parseStmtList();

    // expressions
    bool isExpr();
    Value tryExpr();
    Value parseExpr(Prec prec);
    Value parseExpr() { return parseExpr(BOTTOM); }
    Value parsePrefixExpr();
    Value parseInfixExpr(Value aval);
    Value parsePostfixExpr(Value aval);
    Value parsePrimaryExpr();
    Value parseLiteral();
    Value parseLambda();

    // statements
    void parseStmt();
    void parseExprStmt();
    void parseDeclStmt();
    void parseIfElse();
    void parseWhile();
    void parseDoWhile();
    void parseFor();
    void parseBreak();
    void parseContinue();
    void parseReturn();

    /// '{' stmt '}'
    void parseScopeBody();
    /// stmt which is \em not a compound-stmt \em or '{' stmt '}' \em wihout pushing/poping of scopes
    void parseScope();
    /// (push scope) '{' stmt '}' (pop scope)
    void parseCompoundStmt();

    /// helper for condition in if/while/do-while
    Value parseCond(const std::string& what);
private:

    anydsl::World& world() { return emit.world; }

    /// Consume next Token in input stream, fill look-ahead buffer, return consumed Token.
    Token lex();

    int nextId();

    /*
     * data
     */

    Lexer lexer_;       ///< invoked in order to get next token
    Token lookahead_[2];///< LL(2) look ahead
    Emitter emit;       ///< encapsulates AIR construction

    static Type2Prec    preRPrec_; ///< right precedence -- for unary prefix operators
    static Type2BinPrec binPrec_;  ///< left and right precedences -- for binary operators
    static Type2Prec    postLPrec_;///< left precedence -- for unary postfix operators

    anydsl::BB* break_;   ///< current break target
    anydsl::BB* continue_;///< current continue target

    int counter_;
};

} // namespace impala

#endif // IMPALA_PARSER_H
