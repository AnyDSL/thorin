#include <impala/token.h>

#include "anydsl/util/assert.h"
#include "anydsl/util/cast.h"
#include "anydsl/util/stdlib.h"

using anydsl::HasLocation;
using anydsl::Position;
using anydsl::Location;
using anydsl::Symbol;

namespace impala {

/*
 * constructors
 */

Token::Token(const anydsl::Location& loc, Type tok)
    : HasLocation(loc)
    , symbol_(tok2sym_[tok])
    , type_(tok)
{}

Token::Token(const anydsl::Location& loc, const std::string& str)
    : HasLocation(loc)
    , symbol_( anydsl::SymbolTable::This().get(str) )
{
    Sym2Tok::const_iterator i = keywords_.find(str);
    if (i == keywords_.end())
        type_ = Token::ID;
    else {
        type_ = i->second;
    }
}

Token::Token(const anydsl::Location& loc, Type type, const std::string& str)
    : HasLocation(loc)
    , symbol_( anydsl::SymbolTable::This().get(str) )
    , type_(type) 
{
    using namespace std;

    switch (type_) {
        //case LIT_INT8:   box_.u8_  = anydsl::bcast< uint8_t,  int8_t>( int8_t(strtol  (symbol_.str(), 0, 0))); break;
        //case LIT_INT16:  box_.u16_ = anydsl::bcast<uint16_t, int16_t>(int16_t(strtol  (symbol_.str(), 0, 0))); break;
        //case LIT_INT32:  box_.u32_ = anydsl::bcast<uint32_t, int32_t>(int32_t(strtol  (symbol_.str(), 0, 0)));  break;
        //case LIT_INT64:  box_.u64_ = anydsl::bcast<uint64_t, int64_t>(int64_t(strtoll (symbol_.str(), 0, 0)));  break;

        case LIT_UINT8:  box_. u8_ = uint8_t (strtoul (symbol_.str(), 0, 0)); break;
        case LIT_UINT16: box_.u16_ = uint16_t(strtoul (symbol_.str(), 0, 0)); break;
        case LIT_UINT32: box_.u32_ = uint32_t(strtoul (symbol_.str(), 0, 0)); break;
        case LIT_UINT64: box_.u64_ = uint64_t(strtoull(symbol_.str(), 0, 0)); break;

        case LIT_FLOAT:  box_.f32_ = strtof(symbol_.str(), 0); break;
        case LIT_DOUBLE: box_.f64_ = strtod(symbol_.str(), 0); break;

        default: ANYDSL_UNREACHABLE;
    }
}

bool Token::isArith() const {
    switch (type_) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case MOD: return true;
        default:  return false;
    }
}

bool Token::isRel() const {
    switch (type_) {
        case EQ:
        case NE:
        case LT: 
        case LE: 
        case GT: 
        case GE: return true;
        default: return false;
    }
}

Token Token::seperateAssign() const {
    anydsl_assert(isAsgn(), "must be an assignment other than ASGN");

    switch (type_) {
        case ADD_ASGN: return Token(loc_, ADD);
        case SUB_ASGN: return Token(loc_, SUB);
        case MUL_ASGN: return Token(loc_, MUL);
        case DIV_ASGN: return Token(loc_, DIV);
        case MOD_ASGN: return Token(loc_, MOD);
        case AND_ASGN: return Token(loc_, AND);
        case  OR_ASGN: return Token(loc_,  OR);
        case XOR_ASGN: return Token(loc_, XOR);
        case SHL_ASGN: return Token(loc_, SHL);
        case SHR_ASGN: return Token(loc_, SHR);
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::ArithOpKind Token::toArithOp() const {
    switch (type_) {
        case INC:
        case ADD: return anydsl::ArithOp_add;
        case DEC:
        case SUB: return anydsl::ArithOp_sub;
        case MUL: return anydsl::ArithOp_mul;
        case DIV: return anydsl::ArithOp_udiv;
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::RelOpKind Token::toRelOp() const {
    switch (type_) {
        case EQ: return anydsl::RelOp_cmp_eq;
        case NE: return anydsl::RelOp_cmp_ne;
        case LT: return anydsl::RelOp_cmp_ult;
        case LE: return anydsl::RelOp_cmp_ule;
        case GT: return anydsl::RelOp_cmp_ugt;
        case GE: return anydsl::RelOp_cmp_uge;
        default: ANYDSL_UNREACHABLE;
    }
}

anydsl::PrimTypeKind Token::toPrimType() const {
    switch (type_) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_ ## itype: return anydsl::PrimType_##atype;
#include <impala/tokenlist.h>
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

/*static*/ void Token::init() {
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
#define IMPALA_LIT(       tok, t)         tok2str_[ tok ] = Symbol("<literal>").index();
#define IMPALA_KEY_EXPR(  tok, str)       insertKey(tok, str);
#define IMPALA_KEY_STMT(  tok, str)       insertKey(tok, str);
#define IMPALA_TYPE(itype, atype)         insertKey(TYPE_ ## itype, #itype );
#include <impala/tokenlist.h>

    insert(ERROR, "error");
    insertKey(DEF, "def");

    tok2str_[ID]  =         Symbol("<identifier>").index();
    tok2str_[DEF] =         Symbol("def").index();
    tok2str_[END_OF_FILE] = Symbol("<end of file>").index();
}

/*static*/ void Token::insertKey(TokenType tok, const char* str) {
    Symbol s = str;
    anydsl_assert(keywords_.find(s) == keywords_.end(), "already inserted");
    keywords_[s] = tok;
    tok2str_ [tok] = s.index();
}


/*static*/ Symbol Token::insert(TokenType tok, const char* str) {
    Symbol s = str;
    std::pair<Tok2Sym::iterator, bool> p = tok2sym_.insert( std::make_pair(tok, s) );
#ifndef NDEBUG
    if (!p.second) {
        Type   oldTok = p.first->first;
        Symbol oldSym = p.first->second;
        anydsl_assert(s == oldSym && tok == oldTok, "inserted ambigous duplicate");
    }
#endif

    tok2str_[tok] = s.index();

    return p.first->second;
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TokenType& type) {
    Token::Tok2Str::iterator i = Token::tok2str_.find(type);
    anydsl_assert(i != Token::tok2str_.end(), "must be found");
    return os << Symbol(i->second).str();
}

std::ostream& operator << (std::ostream& os, const Token& tok) {
    const char* sym = tok.symbol().str();
    if (std::strcmp(sym, "") == 0)
        return os << Symbol(Token::tok2str_[tok.type()]).str();
    else
        return os << sym;
}

//------------------------------------------------------------------------------

} // namespace impala
