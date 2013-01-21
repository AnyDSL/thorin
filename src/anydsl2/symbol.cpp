#include "anydsl2/symbol.h"

#include <iomanip>
#include <sstream>

#include "anydsl2/util/for_all.h"
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

void Symbol::insert(const char* s) {
    Table::iterator i = table_.find(s);

    if (i == table_.end())
        i = table_.insert(strdup(s)).first;

    str_ = *i;
}

void Symbol::destroy() {
    for_all (s, table_)
        free((void*) const_cast<char*>(s));
}

} // namespace anydsl2
