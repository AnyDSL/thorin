#ifndef THORIN_UTIL_SYMBOL_H
#define THORIN_UTIL_SYMBOL_H

#include <string>

#include "thorin/util/hash.h"

namespace thorin {

class Symbol {
public:
    struct Hash {
        static uint64_t hash(Symbol s) { return thorin::hash(s.c_str()); }
        static bool eq(Symbol s1, Symbol s2) { return s1 == s2; }
        static Symbol sentinel() { return Symbol(/*dummy*/23); }
    };

    Symbol() { insert(""); }
    Symbol(const char* str) { insert(str); }
    Symbol(const std::string& str) { insert(str.c_str()); }

    const char* c_str() const { return str_; }
    std::string str() const { return str_; }
    operator bool() const { return *this != Symbol(""); }
    bool operator==(Symbol symbol) const { return c_str() == symbol.c_str(); }
    bool operator!=(Symbol symbol) const { return c_str() != symbol.c_str(); }
    bool operator==(const char* s) const { return c_str() == Symbol(s).c_str(); }
    bool operator!=(const char* s) const { return c_str() != Symbol(s).c_str(); }
    bool empty() const { return *str_ == '\0'; }
    bool is_anonymous() { return (*this) == "_"; }
    std::string remove_quotation() const;

private:
    Symbol(int /* just a dummy */)
        : str_((const char*)(1))
    {}

    struct Table {
        ~Table() {
            for (auto s : map)
                free((void*) const_cast<char*>(s));
        }

        HashSet<const char*, StrHash> map;
    };

    void insert(const char* str);

    const char* str_;
    static Table table_;
};

inline Symbol operator+(Symbol s1, Symbol s2) { return std::string(s1.c_str()) + s2.str(); }
inline Symbol operator+(Symbol s1, const char* s2) { return std::string(s1.c_str()) + s2; }
inline Symbol operator+(Symbol s1, std::string s2) { return std::string(s1.c_str()) + s2; }
inline std::ostream& operator<<(std::ostream& os, Symbol s) { return os << s.c_str(); }

}

#endif
