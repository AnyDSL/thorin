#ifndef THORIN_UTIL_VSTREAMF_H
#define THORIN_UTIL_VSTREAMF_H

#include <cstdarg>
#include <ostream>

namespace thorin {

void vstreamf(std::ostream& out, char const* fmt, va_list ap);

class Streamable {
public:
    virtual std::ostream& stream(std::ostream&) const = 0;
    void dump() const;
};

}

#endif
