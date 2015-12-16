#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <iomanip>
#include <ostream>

#include "thorin/util/stream.h"

namespace thorin {

class Log {
    Log() = delete;
    Log(const Log&) = delete;
    Log& operator= (const Log&) = delete;

public:
    enum Level {
        Error, Warn, Info, Debug
    };

    static std::ostream& stream() { return *stream_; }
    static void set(Level max_level, std::ostream* stream, bool print_loc = true) { set_max_level(max_level); set_stream(stream); set_print_loc(print_loc); }
    static Level max_level() { return max_level_; }
    static void set_stream(std::ostream* stream) { stream_ = stream; }
    static void set_max_level(Level max_level) { max_level_ = max_level; }
    static void set_print_loc(bool print_loc) { print_loc_ = print_loc; }
    static char level2char(Level);

    template<typename... Args>
    static void log(Level level, const char* file, int line, const char* fmt, Args... args) {
        if (Log::stream_ && level <= Log::max_level()) {
            if (print_loc_)
                Log::stream() << level2char(level) << ':' << file << ':' << std::setw(4) << line << ": ";
            if (level == Debug)
                Log::stream() << "  ";
            streamf(Log::stream(), fmt, args...);
            Log::stream() << std::endl;
            if (level == Error)
                exit(EXIT_FAILURE);
        }
    }

private:
    static std::ostream* stream_;
    static Level max_level_;
    static bool print_loc_;
};

}

#ifndef NDEBUG
#define LOG(level, ...) thorin::Log::log((level), __FILE__, __LINE__, __VA_ARGS__)
#else
#define LOG(level, ...) do {} while (false)
#endif

#define ELOG(...) LOG(thorin::Log::Error, __VA_ARGS__)
#define WLOG(...) LOG(thorin::Log::Warn,  __VA_ARGS__)
#define ILOG(...) LOG(thorin::Log::Info,  __VA_ARGS__)
#define DLOG(...) LOG(thorin::Log::Debug, __VA_ARGS__)
#define ILOG_SCOPE(s) { \
    ILOG("*** BEGIN: " #s " {"); \
    (s); \
    ILOG("}"); \
}

#endif
