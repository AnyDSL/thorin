#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <iomanip>
#include <cstdlib>
#include <ostream>

#include "thorin/util/stream.h"

namespace thorin {

class Log {
    Log() = delete;
    Log(const Log&) = delete;
    Log& operator= (const Log&) = delete;

public:
    enum Level {
        Debug, Verbose, Info, Warn, Error,
    };

    static std::ostream& stream();
    static void set(Level min_level, std::ostream* stream, bool print_loc = true);
    static Level min_level();
    static void set_stream(std::ostream* stream);
    static void set_min_level(Level min_level);
    static void set_print_loc(bool print_loc);
    static std::string level2string(Level);
    static int level2color(Level);
    static std::string colorize(const std::string&, int);

    template<typename... Args>
    static void log(Level level, const char* file, int line, const char* fmt, Args... args) {
        if (Log::get_stream() && Log::get_min_level() <= level) {
            if (Log::get_print_loc())
                Log::stream() << colorize(level2string(level), level2color(level)) << ':'
                              << colorize(file, 7) << ':' << std::setw(4) << line << ": ";
            if (level == Debug)
                Log::stream() << "  ";
            streamf(Log::stream(), fmt, args...);
            Log::stream() << std::endl;
        }
    }

    template<typename... Args>
    [[noreturn]] static void error(const char* file, int line, const char* fmt, Args... args) {
        log(Error, file, line, fmt, args...);
        std::abort();
    }

private:
    static std::ostream* get_stream();
    static Level get_min_level();
    static bool get_print_loc();

    static std::ostream* stream_;
    static Level min_level_;
    static bool print_loc_;
};

}

#define ALWAYS_LOG(level, ...) thorin::Log::log((level), __FILE__, __LINE__, __VA_ARGS__)
#ifndef NDEBUG
#define MAYBE_LOG(level, ...) ALWAYS_LOG(level, __VA_ARGS__)
#else
#define MAYBE_LOG(level, ...) do {} while (false)
#endif

#define ELOG(...) thorin::Log::error(__FILE__, __LINE__, __VA_ARGS__)
#define WLOG(...) ALWAYS_LOG(thorin::Log::Warn,   __VA_ARGS__)
#define ILOG(...) ALWAYS_LOG(thorin::Log::Info,   __VA_ARGS__)
#define VLOG(...) MAYBE_LOG(thorin::Log::Verbose, __VA_ARGS__)
#define DLOG(...) MAYBE_LOG(thorin::Log::Debug,   __VA_ARGS__)
#define VLOG_SCOPE(s) { \
    VLOG("*** BEGIN: " #s " {"); \
    (s); \
    VLOG("}"); \
}

#endif
