#ifndef THORIN_SYMBOL_H
#define THORIN_SYMBOL_H

#include <cstring>
#include <string>

#include "thorin/util/hash.h"

namespace thorin {

struct StrHash { size_t operator () (const char* s) const; };
struct StrEqual { bool operator () (const char* s1, const char* s2) const { return std::strcmp(s1, s2) == 0; } };

class Symbol {
public:
    Symbol() { insert(""); }
    Symbol(const char* str) { insert(str); }
    Symbol(const std::string& str) { insert(str.c_str()); }

    const char* str() const { return str_; }
    bool operator == (Symbol symbol) const { return str() == symbol.str(); }
    bool operator != (Symbol symbol) const { return str() != symbol.str(); }

    static void destroy();

private:
    void insert(const char* str);

    const char* str_;
    typedef HashSet<const char*, StrHash, StrEqual> Table;
    static Table table_;
};

inline std::ostream& operator << (std::ostream& o, Symbol s) { return o << s.str(); }

template<>
struct Hash<Symbol> {
    size_t operator () (Symbol symbol) const { return hash_value(symbol.str()); }
};

}

namespace std {

template<>
struct hash<thorin::Symbol> {
    size_t operator () (thorin::Symbol symbol) const { return thorin::hash_value(symbol.str()); }
};

}

#endif
