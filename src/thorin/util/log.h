#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <iomanip>
#include <ostream>

#include "streamf.h"

namespace thorin {

class Log {
    Log() = delete;
    Log(const Log&) = delete;
    Log& operator= (const Log&) = delete;

public:
    enum Level {
        Info, Debug
    };

    static std::ostream& stream() { return *stream_; }
    static void set(Level max_level, std::ostream* stream) { set_max_level(max_level); set_stream(stream); }
    static Level max_level() { return max_level_; }
    static void set_stream(std::ostream* stream) { stream_ = stream; }
    static void set_max_level(Level max_level) { max_level_ = max_level; }

    template<typename... Args>
    static void log(Level level, const char* file, int line, const char* fmt, Args... args) {
        if (Log::stream_ && level <= Log::max_level()) {
            Log::stream() << (level == Log::Info ? "I:" : "D:") << file << ':' << std::setw(4) << line << ": ";
            streamf(Log::stream(), fmt, args...);
            Log::stream() << std::endl;
        }
    }

private:
    static std::ostream* stream_;
    static Level max_level_;
};

}

#ifndef NDEBUG
#define LOG(level, ...) thorin::Log::log((level), __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG(level, ...) {}
#endif

#define ILOG(...) LOG(thorin::Log::Info,  __VA_ARGS__)
#define DLOG(...) LOG(thorin::Log::Debug, __VA_ARGS__)

#endif
