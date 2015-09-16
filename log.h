#include "printf.h"

enum LOG_LEVEL {
  DEBUG, INFO
};

class Logging {
public:
	static LOG_LEVEL level;
};

#ifdef LOGGING
#define LOG(_level, format, ...) { 			\
	if(Logging::level <= _level) {			\
		_printf(format, ## __VA_ARGS__);		\
	}						\
}
#else
#define LOG(_level, format, ...)
#endif
