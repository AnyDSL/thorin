#include "anydsl2/symbol.h"

#include <iomanip>
#include <sstream>

#include "anydsl2/util/for_all.h"

namespace anydsl2 {

size_t StrHash::operator () (const char* s) const {
    size_t seed = 0;
    const char* i = s;

    while (*i != '\0')
        boost::hash_combine(seed, *i++);

    boost::hash_combine(seed, i-s);

    return seed;
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

std::string make_name(const char* cstr, int id) {
    std::ostringstream oss;
    oss << '<' << cstr << '-';
    oss << std::setw(2) << std::setfill('0') << id << '>';

    return oss.str();
}

} // namespace anydsl2
