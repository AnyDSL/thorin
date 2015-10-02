#ifndef THORIN_UTIL_VSTREAMF_H
#define THORIN_UTIL_VSTREAMF_H

#include <cstdarg>
#include <ostream>

namespace thorin {

/** 
 * @brief vfprintf-like functions which works on a C++-stream.
 * Additionally, vstreamf allows the following conversion specifiers:
 * * Use '%S' for a @p std::string.
 * * Use '%Y' for a @p Streamable*.
 */
void vstreamf(std::ostream& out, char const* fmt, va_list ap);

/// Inherit from this class and implement @p stream in order to use the '%Y' conversion specifier of @p vstreamf.
class Streamable {
public:
    virtual std::ostream& stream(std::ostream&) const = 0;
    /// Uses @p stream in order to dump to @p std::cout.
    void dump() const;
};

/// Use @p Streamable in C++ streams via operator '<<'.
std::ostream& operator << (std::ostream&, const Streamable*);

}

#endif
