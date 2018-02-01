#include "thorin/util/symbol.h"

#include <iomanip>
#include <sstream>

namespace thorin {

Symbol::Table Symbol::table_;

#ifdef _MSC_VER
static const char* duplicate(const char* s) { return _strdup(s); }
#else // _MSC_VER
static const char* duplicate(const char* s) { return strdup(s); }
#endif // _MSC_VER

void Symbol::insert(const char* s) {
    auto i = table_.map.find(s);
    if (i == table_.map.end())
        i = table_.map.emplace(duplicate(s)).first;
    str_ = *i;
}

std::string Symbol::remove_quotation() const {
    std::string str = str_;
    if (!str.empty() && str.front() == '"') {
        assert(str.size() >= 2 && str.back() == '"');
        str = str.substr(1, str.size()-2);
    }
    return str;
}

}
