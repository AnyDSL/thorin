#include "anydsl/support/symbol.h"

#include "anydsl/util/assert.h"

namespace anydsl {

/*
 * constructor
 */

SymbolTable::SymbolTable() {
    ANYDSL_CALL_ONCE;
    symbolSet_.insert(Symbol()); //add empty string into the map
}

/*
 * further methods
 */

Symbol SymbolTable::get(const char* str) {
    if (str[0]==0) return Symbol(); //empty string
    Symbol tmp = Symbol((size_t)str); //create a temporary Symbol which does not copy str into internal memory

    std::pair<SymbolSet::iterator, bool> res = symbolSet_.insert(tmp);
    if (!res.second)
        return *res.first;// there is already a symbol with the same string

    symbolSet_.erase(res.first); //remove the temporary Symbol
    Symbol result = store(str);
    bool inserted=symbolSet_.insert(result).second; // insert the real Symbol
    assert(inserted);

    return result;
}

Symbol SymbolTable::store(const char* str) {
    size_t strLength = strlen(str);
    assert( strLength>0 ); //we already added the empty string
    char *dest=memory_.alloc(strLength); //it will alloc the extra byte for \0 on its own
    assert(dest);
    memcpy(dest, str, strLength+1);
    return Symbol((size_t)dest);
}

std::ostream& operator << (std::ostream& o, Symbol s) {
    return o << s.str();
}

} // namespace anydsl
