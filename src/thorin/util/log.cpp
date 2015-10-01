#include "log.h"

#include <iomanip>

#include "vstreamf.h"

namespace thorin {

Log::Level Log::max_level_ = Log::Info;
std::ostream* Log::stream_ = nullptr;

void Log::log(Log::Level level, const char* file, int line, const char* fmt, ...) {
	if (Log::stream_ && level <= Log::max_level()) {
        Log::stream() << (level == Log::Info ? "I:" : "D:") << file << ':' 
                      << std::setw(4) << std::setfill('0') << line << ": ";
		va_list ap;
		va_start(ap, fmt);
		vstreamf(Log::stream(), fmt, ap);
		va_end(ap);
        Log::stream() << std::endl;
	}
}

}
