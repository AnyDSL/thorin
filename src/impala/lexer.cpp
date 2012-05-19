#include "impala/lexer.h"

#include <cctype>
#include <cstdio>
#include <stdexcept>

#include "anydsl/support/symbol.h"

using anydsl::Symbol;

/*
 * helpers
 */

namespace {

inline bool sym(int c) { return std::isalpha(c) || c == '_'; }
inline bool dec(int c) { return std::isdigit(c); }
inline bool dec_nonzero(int c) { return c >= '1' && c <= '9'; }
inline bool space(int c) { return std::isspace(c); }
inline bool oct(int c) { return '0' <= c && c <= '7'; }
inline bool hex(int c) { return std::isxdigit(c); }

inline bool bB(int c) { return c == 'b' || c == 'B'; }
inline bool eE(int c) { return c == 'e' || c == 'E'; }
inline bool fF(int c) { return c == 'f' || c == 'F'; }
inline bool lL(int c) { return c == 'l' || c == 'L'; }
inline bool oO(int c) { return c == 'o' || c == 'O'; }
inline bool pP(int c) { return c == 'p' || c == 'P'; }
inline bool sS(int c) { return c == 's' || c == 'S'; }
inline bool uU(int c) { return c == 'u' || c == 'U'; }
inline bool xX(int c) { return c == 'x' || c == 'X'; }
inline bool sgn(int c){ return c == '+' || c == '-'; }
inline bool _89(int c){ return c == '8' || c == '9'; }

} // anonymous namespace

namespace impala {

/*
 * constructor
 */

Lexer::Lexer(std::istream& stream, const std::string& filename)
    : stream_(stream)
    , pos_(filename, 1, 1)
    , loc_(pos_) 
{
    if (!stream_)
        throw std::runtime_error("stream is bad");

    stream_.exceptions(std::istream::badbit);
}

/*
 * lexing
 */

int Lexer::next() {
    int c = stream_.get();

    loc_.setPos2(pos_);

    if (c == '\n') {
        pos_.incLine();
        pos_.resetCol();
    } else if (c != std::istream::traits_type::eof())
        pos_.incCol();

    return c;
}

Token Lexer::lex() {
    while (true) {
        std::string str; // the token string is concatenated here
        bool floating = false;
        loc_.setPos1(pos_);

        // end of file
        if (accept(std::istream::traits_type::eof()))
            return Token(loc_, Token::END_OF_FILE);

        // skip whitespace
        if (accept(space)) {
            while (accept(space)) {}
            continue;
        }

        // +, ++, +=
        if (accept('+')) {
            if ( accept('+') ) return Token(loc_, Token::INC);
            if ( accept('=') ) return Token(loc_, Token::ADD_ASGN);
            return Token(loc_, Token::ADD);
        }

        // -, --, -=, ->
        if (accept('-')) {
            if ( accept('-') ) return Token(loc_, Token::DEC);
            if ( accept('=') ) return Token(loc_, Token::SUB_ASGN);
            if ( accept('>') ) return Token(loc_, Token::ARROW);
            return Token(loc_, Token::SUB);
        }

        // *, *=, %, %=, ^, ^=, =, ==, !, !=, :, :=
#define IMPALA_LEX_OP(op, tok1, tok2) \
        if (accept( op )) { \
            if ( accept('=') ) return Token(loc_, Token:: tok2); \
            return Token(loc_, Token:: tok1); \
        }
        IMPALA_LEX_OP('*', MUL, MUL_ASGN)
        IMPALA_LEX_OP('%', MOD, MOD_ASGN)
        IMPALA_LEX_OP('^', XOR, XOR_ASGN)
        IMPALA_LEX_OP('=', ASGN, EQ)
        IMPALA_LEX_OP('!', L_N,  NE)
        IMPALA_LEX_OP(':', COLON,  COLONEQ)

        // <, <=, <<, <<=, >, >=, >>, >>=
#define IMPALA_LEX_REL_SHIFT(op, tok_rel, tok_rel_eq, tok_shift, tok_shift_asgn) \
        if (accept( op )) { \
            if ( accept('=') ) return Token(loc_, Token:: tok_rel_eq); \
            if ( accept(op) ) \
            {  \
                if ( accept('=') ) return Token(loc_, Token:: tok_shift_asgn); \
                return Token(loc_, Token:: tok_shift); \
            } \
            return Token(loc_, Token:: tok_rel); \
        }
        IMPALA_LEX_REL_SHIFT('<', LT, LE, SHL, SHL_ASGN)
        IMPALA_LEX_REL_SHIFT('>', GT, GE, SHR, SHR_ASGN)

        // /, /=, comments
#define IMPALA_WITHIN_COMMENT(delim) \
        while (true) { \
            if (accept(std::istream::traits_type::eof())) { \
                loc_.pos1().error() << "unterminated comment\n"; \
                return Token(loc_, Token::END_OF_FILE); \
            } \
            if (delim) break; \
            next(); /* eat up char in comment */\
        }
        if ( accept('/') ) {
            if ( accept('=') )
                return Token(loc_, Token::DIV_ASGN);
            if ( accept('*') ) { // arbitrary comment
                IMPALA_WITHIN_COMMENT(accept('*') && accept('/'));
                continue;
            } 
            if ( accept('/') ) { // end of line comment
                IMPALA_WITHIN_COMMENT(accept('\n'));
                continue;
            }
            return Token(loc_, Token::DIV);
        }

        // &, &=, &&, |, |=, ||
#define IMPALA_LEX_AND_OR(op, tok_bit, tok_logic, tok_asgn) \
        if (accept( op )) { \
            if ( accept('=') ) \
                return Token(loc_, Token:: tok_asgn); \
            if ( accept(op) ) \
                return Token(loc_, Token:: tok_logic); \
            return Token(loc_, Token:: tok_bit); \
        }
        IMPALA_LEX_AND_OR('&', AND, L_A, AND_ASGN)
        IMPALA_LEX_AND_OR('|',  OR, L_O,  OR_ASGN)

        // single character tokens
        if (accept(',')) return Token(loc_, Token::COMMA);
        if (accept(';')) return Token(loc_, Token::SEMICOLON);
        if (accept('(')) return Token(loc_, Token::L_PAREN);
        if (accept(')')) return Token(loc_, Token::R_PAREN);
        if (accept('[')) return Token(loc_, Token::L_BRACKET);
        if (accept(']')) return Token(loc_, Token::R_BRACKET);
        if (accept('{')) return Token(loc_, Token::L_BRACE);
        if (accept('}')) return Token(loc_, Token::R_BRACE);
        if (accept('~')) return Token(loc_, Token::NOT);
        if (accept('?')) return Token(loc_, Token::QUESTION_MARK);

        // '.', floats
        if (accept('.')) {
            std::string str(1, '.');
            if (accept(str, dec)) goto l_fractional_dot;
            return Token(loc_, Token::DOT);
        }

        // identifiers/keywords
        if (accept(str, sym)) {
            while ( accept(str, sym) || accept(str, dec) ) {}

            return Token(loc_, str);
        }

        /*
         * literals
         */

        if (accept(str, '0')) goto l_0;
        if (accept(str, dec_nonzero)) goto l_decimal;

        // invalid input char
        pos_.error() << "invalid input character '" << (char) next() << "'\n";
        continue;

l_0: // 0
        if (accept(str, '.')) goto l_fractional_dot;
        if (accept(str, oct)) goto l_octal;
        if (accept(str, _89)) goto l_fractional;
        if (accept(str,  eE)) goto l_exp;
        if (accept(str,  xX)) goto l_0x;
        goto l_out;

l_0x: // 0x
        if (accept(str, '.')) goto l_0x_dot;
        if (accept(str, hex)) goto l_hexadecimal;
        goto l_error;

l_0x_dot: // 0x.
		if (accept(str, hex)) goto l_hex_dot;
        goto l_error;

l_decimal: // [1-9][0-9]*
		if (accept(str, '.')) goto l_fractional_dot;
		if (accept(str, dec)) goto l_decimal;
		if (accept(str,  eE)) goto l_exp;
        goto l_out;

l_exp: // ...[eE]|[pP]
        floating = true;
        accept(str, sgn);
        if (accept(str, dec)) {
            while (accept(str, dec)) {}
            goto l_out;
        }
        goto l_error;

l_fractional: // 0[0-7]*[89][0-9]*
		while (accept(str, dec)) {}
		if (accept(str, '.')) goto l_fractional_dot;
		if (accept(str,  eE)) goto l_exp;
        goto l_error;

l_fractional_dot: // [0-9]+.[0-9]* .[0-9]+
		while (accept(str, dec)) {}
		if (accept(str,  eE)) goto l_exp;
        floating = true;
        goto l_out;

l_hex_dot: // 0x[0-9A-Fa-f]+.[0-9A-Fa-f]*  0x.[0-9A-Fa-f]+
		while (accept(str, hex)) {}
		if (accept(str,  pP)) goto l_exp;
        goto l_error;

l_hexadecimal: // 0[Xx][0-9A-Fa-f]+
		while (accept(str, hex)) {}
		if (accept(str, '.')) goto l_hex_dot;
		if (accept(str,  pP)) goto l_exp;
        goto l_out;

l_octal: // 0[0-7]+
		while (accept(str, oct)) {}
		if (accept(str, '.')) goto l_fractional_dot;
		if (accept(str, _89)) goto l_fractional;
		if (accept(str,  eE)) goto l_exp;
        goto l_out;

l_error: 
        pos_.error() << "invalid constant '" << str << "'\n";
        // fall through intended
l_out:
        // lex suffix
        TokenKind tok = floating ? Token::LIT_DOUBLE : Token::LIT_INT32;
        
        if (floating) {
            if (accept(str, fF))
                tok = Token::LIT_FLOAT;
        } else {
            if (accept(str, uU)) {
                     if (accept(str, bB)) tok = Token:: LIT_UINT8;
                else if (accept(str, sS)) tok = Token::LIT_UINT16;
                else if (accept(str, lL)) tok = Token::LIT_UINT64;
            } else {
                     if (accept(str, bB)) tok = Token:: LIT_INT8;
                else if (accept(str, sS)) tok = Token::LIT_INT16;
                else if (accept(str, lL)) tok = Token::LIT_INT64;
            }
        }

        // eat up erroneous trailing suffixing chars
        if (accept(str, sym)) {
            while (accept(str, sym)) {}
            loc_.error() << "invalid suffix on " << (floating ? "floating" : "integer") << " constant '" << str << "'\n";
        }

        return Token(loc_, tok, str);
    }
}

} // namespace impala
