#include "impala/token.h"

#include "anydsl2/util/assert.h"
#include "anydsl2/util/cast.h"
#include "anydsl2/util/stdlib.h"

using namespace anydsl2;

namespace impala {

/*
 * constructors
 */

Token::Token(const Location& loc, Kind tok)
    : HasLocation(loc)
    , symbol_(tok2sym_[tok])
    , kind_(tok)
{}

Token::Token(const Location& loc, const std::string& str)
    : HasLocation(loc)
    , symbol_(str)
{
    auto i = keywords_.find(str);
    if (i == keywords_.end())
        kind_ = Token::ID;
    else
        kind_ = i->second;
}

Token::Token(const Location& loc, Kind kind, const std::string& str)
    : HasLocation(loc)
    , symbol_(str)
    , kind_(kind) 
{
    using namespace std;

    switch (kind_) {
        case LIT_int8:   box_ = Box(bcast< uint8_t,  int8_t>( int8_t(strtol  (symbol_.str(), 0, 0)))); break;
        case LIT_int16:  box_ = Box(bcast<uint16_t, int16_t>(int16_t(strtol  (symbol_.str(), 0, 0)))); break;
        case LIT_int32:  box_ = Box(bcast<uint32_t, int32_t>(int32_t(strtol  (symbol_.str(), 0, 0)))); break;
        case LIT_int64:  box_ = Box(bcast<uint64_t, int64_t>(int64_t(strtoll (symbol_.str(), 0, 0)))); break;

        //case LIT_uint8:  box_ = Box(uint8_t (strtoul (symbol_.str(), 0, 0))); break;
        //case LIT_uint16: box_ = Box(uint16_t(strtoul (symbol_.str(), 0, 0))); break;
        //case LIT_uint32: box_ = Box(uint32_t(strtoul (symbol_.str(), 0, 0))); break;
        //case LIT_uint64: box_ = Box(uint64_t(strtoull(symbol_.str(), 0, 0))); break;

        case LIT_float:  box_ = Box(strtof(symbol_.str(), 0)); break;
        case LIT_double: box_ = Box(strtod(symbol_.str(), 0)); break;

        default: ANYDSL2_UNREACHABLE;
    }
}

/*static */ bool Token::is_rel(Kind op) {
    switch (op) {
        case EQ: case LT: case LE: 
        case NE: case GT: case GE: return true;
        default: return false;
    }
}

TokenKind Token::separate_assign(TokenKind kind) {
    assert(is_assign(kind) && "must be an assignment other than ASGN");

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
        default: ANYDSL2_UNREACHABLE;
    }
}

int Token::to_binop(Kind kind, bool is_float) {
    switch (kind) {
        case INC:
        case ADD: return is_float ? ArithOp_fadd : ArithOp_add;
        case DEC:
        case SUB: return is_float ? ArithOp_fsub : ArithOp_sub ;
        case MUL: return is_float ? ArithOp_fmul : ArithOp_mul ;
        case DIV: return is_float ? ArithOp_fdiv : ArithOp_sdiv;
        case MOD: return is_float ? ArithOp_frem : ArithOp_srem;
        case  EQ: return is_float ? RelOp_fcmp_oeq : RelOp_cmp_eq ;
        case  NE: return is_float ? RelOp_fcmp_one : RelOp_cmp_ne ;
        case  LT: return is_float ? RelOp_fcmp_olt : RelOp_cmp_slt;
        case  LE: return is_float ? RelOp_fcmp_ole : RelOp_cmp_sle;
        case  GT: return is_float ? RelOp_fcmp_ogt : RelOp_cmp_sgt;
        case  GE: return is_float ? RelOp_fcmp_oge : RelOp_cmp_sge;
        case AND: assert(!is_float); return ArithOp_and;
        case  OR: assert(!is_float); return ArithOp_or;
        case XOR: assert(!is_float); return ArithOp_xor;
        case SHL: assert(!is_float); return ArithOp_shl;
        case SHR: assert(!is_float); return ArithOp_ashr;
        default: ANYDSL2_UNREACHABLE;
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
    ANYDSL2_CALL_ONCE;

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
#define IMPALA_KEY_EXPR(  tok, str)       insert_key(tok, str);
#define IMPALA_KEY_STMT(  tok, str)       insert_key(tok, str);
#define IMPALA_KEY_MISC(  tok, str)       insert_key(tok, str);
#define IMPALA_TYPE(itype, atype)         insert_key(TYPE_ ## itype, #itype );
#include "impala/tokenlist.h"

    tok2str_[ID]         = Symbol("<identifier>").str();
    insert(END_OF_FILE, "<end of file>");
}

/*static*/ void Token::insert_key(TokenKind tok, const char* str) {
    Symbol s = str;
    assert(keywords_.find(s) == keywords_.end() && "already inserted");
    keywords_[s] = tok;
    tok2str_ [tok] = s.str();
}


/*static*/ Symbol Token::insert(TokenKind tok, const char* str) {
    Symbol s = str;
    auto p = tok2sym_.insert(std::make_pair(tok, s));

#ifndef NDEBUG
    if (!p.second) {
        Kind   oldTok = p.first->first;
        Symbol oldSym = p.first->second;
        assert(s == oldSym && tok == oldTok && "inserted ambigous duplicate");
    }
#endif

    tok2str_[tok] = s.str();
    return p.first->second;
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TokenKind& kind) {
    auto i = Token::tok2str_.find(kind);
    assert(i != Token::tok2str_.end() && "must be found");
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
