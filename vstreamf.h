#ifndef THORIN_UTIL_VSTREAMF_H
#define THORIN_UTIL_VSTREAMF_H

#include <cstdarg>
#include <ostream>

namespace thorin {

void vstreamf(std::ostream& out, char const *fmt, va_list ap);

}

#endif
