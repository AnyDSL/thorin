#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <cstdarg>
#include <ostream>

namespace thorin {

enum class LogLevel {
    Debug, Info
};

class Logging {
public:
	static LogLevel level;
};

void logvf(LogLevel level, const char* fmt, ...);

class Printable {
public:
    virtual const void print(std::ostream& out) const = 0;
};

}

#ifdef LOGGING
#define LOG(level, ...) logvf((level), __VA_ARGS__)
#else
#define LOG(level, ...)
#endif

#define ILOG(...) LOG(thorin::LogLevel::Info,  __VA_ARGS__)
#define DLOG(...) LOG(thorin::LogLevel::Debug, __VA_ARGS__)

#endif
