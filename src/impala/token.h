#ifndef IMPALA_TOKEN_H
#define IMPALA_TOKEN_H

#include <ostream>
#include <string>

#include <boost/unordered_map.hpp>

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
#define IMPALA_LIT(       tok, t)           LIT_##tok,
#define IMPALA_TYPE(itype, atype)           TYPE_##itype,
#include <impala/tokenlist.h>

        // manually insert missing unary prefix/postfix types
        NOT, L_N, INC, DEC,

        // types that need special handling
        TYPE_int, TYPE_uint, TYPE_void, TYPE_noret,


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

    bool isPrefix()  const { return isPrefix(kind_); }
    bool isInfix()   const { return isInfix(kind_); }
    bool isPostfix() const { return isPostfix(kind_); }
    bool isAsgn()    const { return isAsgn(kind_); }
    bool isOp()      const { return isOp(kind_); }
    bool isArith()   const { return isArith(kind_); }
    bool isRel()     const { return isRel(kind_); }

    static bool isPrefix(Kind kind)  { return tok2op_[kind] &  PREFIX; }
    static bool isInfix(Kind kind)   { return tok2op_[kind] &   INFIX; }
    static bool isPostfix(Kind kind) { return tok2op_[kind] & POSTFIX; }
    static bool isAsgn(Kind kind)    { return tok2op_[kind] & ASGN_OP; }
    static bool isOp(Kind kind)      { return isPrefix(kind) || isInfix(kind) || isPostfix(kind); }
    static bool isArith(Kind kind);
    static bool isRel(Kind kind);
    static Kind seperateAssign(Kind kind);

    static int toBinOp(Kind kind);
    static anydsl::ArithOpKind toArithOp(Kind kind);
    static anydsl::RelOpKind toRelOp(Kind kind);
    static anydsl::PrimTypeKind toPrimType(Kind kind);

    /*
     * comparisons
     */

    bool operator == (const Token& t) const { return kind_ == t; }
    bool operator != (const Token& t) const { return kind_ != t; }

    /*
     * statics
     */

private:

    static void init();

    anydsl::Symbol symbol_;
    Kind kind_;
    anydsl::Box box_;

    static int tok2op_[NUM_TOKENS];
    static anydsl::Symbol insert(Kind tok, const char* str);
    static void insertKey(Kind tok, const char* str);

    typedef boost::unordered_map<Kind, anydsl::Symbol> Tok2Sym;
    static Tok2Sym tok2sym_;

    typedef boost::unordered_map<anydsl::Symbol, Kind> Sym2Tok;
    static Sym2Tok keywords_;

    typedef boost::unordered_map<Kind, const char*> Tok2Str;
    static Tok2Str tok2str_;

    friend void init();
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
