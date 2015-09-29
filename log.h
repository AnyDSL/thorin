#include "diagnostic.h"

#include <iostream>

enum LOG_LEVEL {
  DEBUG, INFO
};

class Logging {
public:
	static LOG_LEVEL level;
};

static void logvf(LOG_LEVEL level, const char *fmt, ...) {
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
