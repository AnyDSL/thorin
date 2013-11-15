#include "anydsl2/util/symbol.h"

#include <iomanip>
#include <sstream>

#include "anydsl2/util/hash.h"

namespace anydsl2 {

size_t StrHash::operator () (const char* s) const {
    size_t seed = 0;
    const char* i = s;

    while (*i != '\0')
        seed = hash_combine(seed, *i++);

    return hash_combine(seed, i-s);
}

Symbol::Table Symbol::table_(1031);

#ifdef _MSC_VER
static const char* duplicate(const char* s) { return _strdup(s); }
#else // _MSC_VER
static const char* duplicate(const char* s) { return strdup(s); }
#endif // _MSC_VER

void Symbol::insert(const char* s) {
    auto i = table_.find(s);
    if (i == table_.end())
        i = table_.insert(duplicate(s)).first;
    str_ = *i;
}

void Symbol::destroy() {
    for (auto s : table_)
        free((void*) const_cast<char*>(s));
}

} // namespace anydsl2
