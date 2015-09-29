#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <cstdarg>
#include <ostream>

namespace thorin {

class Log {
    Log(const Log&) = delete;
    Log& operator= (const Log&) = delete;

public:
    enum Level {
        None, Info, Debug
    };

    static std::ostream& stream() { return *stream_; }
    static void set_stream(std::ostream& stream) { stream_ = &stream; }
    static Level level() { return level_; }
    static void set_level(Level level) { level_ = level; }
    static void log(Level, const char* fmt, ...);

private:
    static std::ostream* stream_;
    static Level level_;
};

class Streamable {
public:
    virtual const void stream(std::ostream&) const = 0;
};

}

#ifdef LOGGING
#define LOG(level, ...) thorin::Log::log((level), __VA_ARGS__)
#else
#define LOG(level, ...)
#endif

#define ILOG(...) LOG(thorin::Log::Info,  __VA_ARGS__)
#define DLOG(...) LOG(thorin::Log::Debug, __VA_ARGS__)

#endif
