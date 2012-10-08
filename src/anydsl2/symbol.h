#ifndef ANYDSL2_SYMBOL_H
#define ANYDSL2_SYMBOL_H

#include <string>
#include <cstring>

#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

namespace anydsl2 {

struct StrHash : std::unary_function<const char*, size_t> {
    size_t operator () (const char* s) const;
};

struct StrEqual : std::binary_function<const char*, const char*, bool> {
    bool operator () (const char* s1, const char* s2) const { return std::strcmp(s1, s2) == 0; }
};

class Symbol {
public:

    Symbol() {}
    Symbol(const char* str) { insert(str); }
    Symbol(const std::string& str) { insert(str.c_str()); }

    const char* str() const { return str_; }

    bool operator == (const Symbol& sym) const { return str() == sym.str(); }
    bool operator != (const Symbol& sym) const { return str() == sym.str(); }

    static void destroy();

private:

    void insert(const char* str);

    const char* str_;

    typedef boost::unordered_set<const char*, StrHash, StrEqual> Table;
    static Table table_;
};

inline std::ostream& operator << (std::ostream& o, Symbol s) { return o << s.str(); }
inline size_t hash_value(const Symbol& symbol) { return boost::hash_value(symbol.str()); }
std::string make_name(const char* cstr, int id);
inline Symbol make_symbol(const char* cstr, int id) { return Symbol(make_name(cstr, id)); }

} // namespace anydsl2

#endif
