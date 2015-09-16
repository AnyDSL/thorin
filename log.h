enum LOG_LEVEL {
  DEBUG, INFO
};

class Logging {
public:
	static LOG_LEVEL level;
};

/*
 *	 Custom Class
 */

class Counter {
public:
	Counter() : c(0) {};

	void inc() {
		c++;
	}

	const int get() const {
		return c;
	}

private:
	int c;
};

#ifdef LOGGING
#define LOG(_level, format, ...) { 			\
	if(Logging::level <= _level) {			\
		printf(format, ## __VA_ARGS__);		\
	}						\
}
#else
#define LOG(_level, format, ...)
#endif
