#ifndef DSLU_LEX_HELPERS_H
#define DSLU_LEX_HELPERS_H

namespace anydsl {

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

} // namespace anydsl

#endif // DSLU_LEX_HELPERS_H
