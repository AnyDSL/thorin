#include "diagnostic.h"

#include <iostream>

enum LOG_LEVEL {
  DEBUG, INFO
};

class Logging {
public:
	static LOG_LEVEL level;
};

static void logvf(LOG_LEVEL _level, const char *fmt, ...) {
	if(Logging::level <= _level) {
		va_list argp;
		va_start(argp, fmt);
	
		messagevf(std::cout, fmt, argp);
		
		va_end(argp);
	}
}

#ifdef LOGGING
#define LOG(_level, format, ...) { 			\
	logvf(_level, format, ## __VA_ARGS__);          \
}
#else
#define LOG(_level, format, ...)
#endif
