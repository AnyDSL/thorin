#ifndef IMPALA_TOKEN_H
#define IMPALA_TOKEN_H

#include <map>
#include <ostream>
#include <string>

#include "anydsl/enums.h"
#include "anydsl/symbol.h"
#include "anydsl/util/box.h"
#include "anydsl/util/assert.h"
#include "anydsl/util/location.h"

namespace impala {

class Token : public anydsl::HasLocation {
public:

    enum Kind {
        /*
         * !!! DO NOT CHANGE THIS ORDER !!!
         */

        // add prefix and postfix tokens manually in order to avoid duplicates in the enum
#define IMPALA_INFIX(     tok, t_str, r, l) tok,
#define IMPALA_INFIX_ASGN(tok, t_str, r, l) tok,
#define IMPALA_KEY_EXPR(  tok, t_str)       tok,
#define IMPALA_KEY_STMT(  tok, t_str)       tok,
#define IMPALA_MISC(      tok, t_str)       tok,
#define IMPALA_LIT(       tok, t)           tok,
#define IMPALA_TYPE(itype, atype)           TYPE_ ## itype,
#include <impala/tokenlist.h>

        // manually insert missing unary prefix/postfix types
        NOT, L_N, INC, DEC,

        ERROR, // for debuggin only

        // these do ont appear in impala/tokenlist.h -- they are too special
        ID, END_OF_FILE, DEF,
        NUM_TOKENS
    };

    /*
     * constructors
     */

    /// Empty default constructor; needed for c++ maps etc
    Token() {}

    /// Create a literal operator or special char token
    Token(const anydsl::Location& loc, Kind tok);

    /// Create an identifier (\p ID) or a keyword
    Token(const anydsl::Location& loc, const std::string& str);

    /// Create a literal
    Token(const anydsl::Location& loc, Kind type, const std::string& str);

    /*
     * getters
     */

    anydsl::Symbol symbol() const { return symbol_; }
    anydsl::Box box() const { return box_; }
    Kind kind() const { return kind_; }
    operator Kind () const { return kind_; }

    /*
     * operator/literal stuff
     */

    enum Op {
        NONE    = 0,
        PREFIX  = 1,
        INFIX   = 2,
        POSTFIX = 4,
        ASGN_OP = 8
    };

    int op() const { return tok2op_[kind_]; }

    bool isPrefix()  const { return op() &  PREFIX; }
    bool isInfix()   const { return op() &   INFIX; }
    bool isPostfix() const { return op() & POSTFIX; }
    bool isAsgn()    const { return op() & ASGN_OP; }
    bool isOp() const { return isPrefix() || isInfix() || isPostfix(); }

    bool isArith() const;
    bool isRel() const;

    Token seperateAssign() const;
    anydsl::ArithOpKind toArithOp() const;
    anydsl::RelOpKind toRelOp() const;
    anydsl::PrimTypeKind toPrimType() const;

    /*
     * comparisons
     */

    bool operator == (const Token& t) const { return kind_ == t; }
    bool operator != (const Token& t) const { return kind_ != t; }

    /*
     * statics
     */

    static void init();

private:

    anydsl::Symbol symbol_;
    Kind kind_;
    anydsl::Box box_;

    static int tok2op_[NUM_TOKENS];
    static anydsl::Symbol insert(Kind tok, const char* str);
    static void insertKey(Kind tok, const char* str);

    typedef std::map<Kind, anydsl::Symbol> Tok2Sym;
    static Tok2Sym tok2sym_;

    typedef std::map<anydsl::Symbol, Kind, anydsl::Symbol::FastLess> Sym2Tok;
    static Sym2Tok keywords_;

    typedef std::map<Kind, size_t> Tok2Str;
    static Tok2Str tok2str_;

    friend std::ostream& operator << (std::ostream& os, const Token& tok);
    friend std::ostream& operator << (std::ostream& os, const Kind&  tok);
};

typedef Token::Kind TokenKind;

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const Token& tok);
std::ostream& operator << (std::ostream& os, const TokenKind& tok);

//------------------------------------------------------------------------------

} // namespace impala

#endif
