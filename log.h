#include "diagnostic.h"

#include <iostream>

enum class LogLevel {
  Debug, Info
};

class Logging {
public:
	static LogLevel level;
};

static void logvf(LogLevel level, const char* fmt, ...) {
	if(Logging::level <= level) {
		va_list argp;
		va_start(argp, fmt);
	
		messagevf(std::cout, fmt, argp);
		
		va_end(argp);
	}
}

#ifdef LOGGING
#define LOG(level, ...) logvf((level), __VA_ARGS__)
#else
#define LOG(level, ...)
#endif

#define ILOG(...) LOG(LogLevel::Info,  __VA_ARGS__)
#define DLOG(...) LOG(LogLevel::Debug, __VA_ARGS__)
