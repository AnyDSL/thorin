#include <impala/token.h>

#include "anydsl/util/assert.h"
#include "anydsl/util/cast.h"
#include "anydsl/util/stdlib.h"

using anydsl::Box;
using anydsl::HasLocation;
using anydsl::Location;
using anydsl::Position;
using anydsl::Symbol;
using anydsl::bcast;

namespace impala {

/*
 * constructors
 */

Token::Token(const anydsl::Location& loc, Kind tok)
    : HasLocation(loc)
    , symbol_(tok2sym_[tok])
    , kind_(tok)
{}

Token::Token(const anydsl::Location& loc, const std::string& str)
    : HasLocation(loc)
    , symbol_(str)
{
    Sym2Tok::const_iterator i = keywords_.find(str);
    if (i == keywords_.end())
        kind_ = Token::ID;
    else
        kind_ = i->second;
}

Token::Token(const anydsl::Location& loc, Kind kind, const std::string& str)
    : HasLocation(loc)
    , symbol_(str)
    , kind_(kind) 
{
    using namespace std;

    switch (kind_) {
        case LIT_int8:   box_ = Box(bcast< uint8_t,  int8_t>( int8_t(strtol  (symbol_.str(), 0, 0)))); break;
        case LIT_int16:  box_ = Box(bcast<uint16_t, int16_t>(int16_t(strtol  (symbol_.str(), 0, 0)))); break;
        case LIT_int32:  box_ = Box(bcast<uint32_t, int32_t>(int32_t(strtol  (symbol_.str(), 0, 0))));  break;
        case LIT_int64:  box_ = Box(bcast<uint64_t, int64_t>(int64_t(strtoll (symbol_.str(), 0, 0))));  break;

        case LIT_uint8:  box_ = Box(uint8_t (strtoul (symbol_.str(), 0, 0))); break;
        case LIT_uint16: box_ = Box(uint16_t(strtoul (symbol_.str(), 0, 0))); break;
        case LIT_uint32: box_ = Box(uint32_t(strtoul (symbol_.str(), 0, 0))); break;
        case LIT_uint64: box_ = Box(uint64_t(strtoull(symbol_.str(), 0, 0))); break;

        case LIT_float:  box_ = Box(strtof(symbol_.str(), 0)); break;
        case LIT_double: box_ = Box(strtod(symbol_.str(), 0)); break;

        default: ANYDSL_UNREACHABLE;
    }
}

bool Token::is_arith(Kind op) {
    switch (op) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case MOD: 
        case AND:
        case  OR:
        case XOR:
            return true;
        default:  return false;
    }
}

bool Token::is_rel(Kind op) {
    switch (op) {
        case EQ:
        case NE:
        case LT: 
        case LE: 
        case GT: 
        case GE: return true;
        default: return false;
    }
}

TokenKind Token::seperateAssign(TokenKind kind) {
    anydsl_assert(is_asgn(kind), "must be an assignment other than ASGN");

    switch (kind) {
        case ADD_ASGN: return ADD;
        case SUB_ASGN: return SUB;
        case MUL_ASGN: return MUL;
        case DIV_ASGN: return DIV;
        case MOD_ASGN: return MOD;
        case AND_ASGN: return AND;
        case  OR_ASGN: return OR;
        case XOR_ASGN: return XOR;
        case SHL_ASGN: return SHL;
        case SHR_ASGN: return SHR;
        default: ANYDSL_UNREACHABLE;
    }
}

int Token::toBinOp(Kind kind) {
    switch (kind) {
        case INC:
        case ADD: return anydsl::ArithOp_add;
        case DEC:
        case SUB: return anydsl::ArithOp_sub;
        case MUL: return anydsl::ArithOp_mul;
        case DIV: return anydsl::ArithOp_udiv;
        case AND: return anydsl::ArithOp_and;
        case  OR: return anydsl::ArithOp_or;
        case XOR: return anydsl::ArithOp_xor;
        case  EQ: return anydsl::RelOp_cmp_eq;
        case  NE: return anydsl::RelOp_cmp_ne;
        case  LT: return anydsl::RelOp_cmp_ult;
        case  LE: return anydsl::RelOp_cmp_ule;
        case  GT: return anydsl::RelOp_cmp_ugt;
        case  GE: return anydsl::RelOp_cmp_uge;
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::ArithOpKind Token::toArithOp(Kind kind) {
    switch (kind) {
        case INC:
        case ADD: return anydsl::ArithOp_add;
        case DEC:
        case SUB: return anydsl::ArithOp_sub;
        case MUL: return anydsl::ArithOp_mul;
        case DIV: return anydsl::ArithOp_udiv;
        case AND: return anydsl::ArithOp_and;
        case  OR: return anydsl::ArithOp_or;
        case XOR: return anydsl::ArithOp_xor;
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::RelOpKind Token::toRelOp(Kind kind) {
    switch (kind) {
        case EQ: return anydsl::RelOp_cmp_eq;
        case NE: return anydsl::RelOp_cmp_ne;
        case LT: return anydsl::RelOp_cmp_ult;
        case LE: return anydsl::RelOp_cmp_ule;
        case GT: return anydsl::RelOp_cmp_ugt;
        case GE: return anydsl::RelOp_cmp_uge;
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::PrimTypeKind Token::toPrimType(Kind kind) {
    switch (kind) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_ ## itype: return anydsl::PrimType_##atype;
#include "impala/tokenlist.h"
        default: ANYDSL_UNREACHABLE;
    }
}

/*
 * static member variables
 */

int Token::tok2op_[NUM_TOKENS];
Token::Sym2Tok Token::keywords_;
Token::Tok2Sym Token::tok2sym_;
Token::Tok2Str Token::tok2str_;

/*
 * static methods
 */

void Token::init() {
    ANYDSL_CALL_ONCE;

    /*
     * - set pre-/in-/postfix operators
     * - register literals
     * - register keywords
     * - register misc tokens
     */

    for (size_t i = 0; i < NUM_TOKENS; ++i)
        tok2op_[i] = NONE;

#define IMPALA_PREFIX(    tok, str, r   ) insert(tok, str); tok2op_[tok] |= PREFIX;  
#define IMPALA_POSTFIX(   tok, str,    l) insert(tok, str); tok2op_[tok] |= POSTFIX; 
#define IMPALA_INFIX(     tok, str, r, l) insert(tok, str); tok2op_[tok] |= INFIX;   
#define IMPALA_INFIX_ASGN(tok, str, r, l) insert(tok, str); tok2op_[tok] |= INFIX | ASGN_OP;
#define IMPALA_MISC(      tok, str)       insert(tok, str);
#define IMPALA_LIT(       tok, atype)     tok2str_[LIT_##tok] = Symbol("<literal>").str();
#define IMPALA_KEY_EXPR(  tok, str)       insertKey(tok, str);
#define IMPALA_KEY_STMT(  tok, str)       insertKey(tok, str);
#define IMPALA_TYPE(itype, atype)         insertKey(TYPE_ ## itype, #itype );
#include <impala/tokenlist.h>

    insertKey(TYPE_int,   "int");
    insertKey(TYPE_uint,  "uint");
    insertKey(TYPE_void,  "void");
    insertKey(TYPE_noret, "noret");
    insertKey(DEF,        "def");

    tok2str_[TYPE_int]   = Symbol("int").str();
    tok2str_[TYPE_uint]  = Symbol("uint").str();
    tok2str_[TYPE_noret] = Symbol("noret").str();
    tok2str_[DEF]        = Symbol("def").str();
    tok2str_[ID]         = Symbol("<identifier>").str();
    tok2str_[END_OF_FILE]= Symbol("<end of file>").str();
}

/*static*/ void Token::insertKey(TokenKind tok, const char* str) {
    Symbol s = str;
    anydsl_assert(keywords_.find(s) == keywords_.end(), "already inserted");
    keywords_[s] = tok;
    tok2str_ [tok] = s.str();
}


/*static*/ Symbol Token::insert(TokenKind tok, const char* str) {
    Symbol s = str;
    std::pair<Tok2Sym::iterator, bool> p = tok2sym_.insert( std::make_pair(tok, s) );

#ifndef NDEBUG
    if (!p.second) {
        Kind   oldTok = p.first->first;
        Symbol oldSym = p.first->second;
        anydsl_assert(s == oldSym && tok == oldTok, "inserted ambigous duplicate");
    }
#endif

    tok2str_[tok] = s.str();

    return p.first->second;
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TokenKind& kind) {
    Token::Tok2Str::iterator i = Token::tok2str_.find(kind);
    anydsl_assert(i != Token::tok2str_.end(), "must be found");
    return os << Symbol(i->second).str();
}

std::ostream& operator << (std::ostream& os, const Token& tok) {
    const char* sym = tok.symbol().str();
    if (std::strcmp(sym, "") == 0)
        return os << Symbol(Token::tok2str_[tok.kind()]).str();
    else
        return os << sym;
}

//------------------------------------------------------------------------------

} // namespace impala
