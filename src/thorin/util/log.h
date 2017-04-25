#ifndef THORIN_UTIL_LOG_H
#define THORIN_UTIL_LOG_H

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <ostream>
#include <sstream>

#include "thorin/util/location.h"
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
    static void log(Level level, Location location, const char* fmt, Args... args) {
        if (Log::get_stream() && Log::get_min_level() <= level) {
            std::ostringstream oss;
            oss << location;
            if (Log::get_print_loc())
                Log::stream() << colorize(level2string(level), level2color(level)) << ':'
                              << colorize(oss.str(), 7) << ": ";
            if (level == Debug)
                Log::stream() << "  ";
            streamf(Log::stream(), fmt, std::forward<Args>(args)...);
            Log::stream() << std::endl;
        }
    }

    template<typename... Args>
    [[noreturn]] static void error(Location location, const char* fmt, Args... args) {
        log(Error, location, fmt, args...);
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

template<typename... Args>
std::ostream& outf(const char* fmt, Args... args) { return streamf(std::cout, fmt, std::forward<Args>(args)...); }
template<typename... Args>
std::ostream& errf(const char* fmt, Args... args) { return streamf(std::cerr, fmt, std::forward<Args>(args)...); }

}

#define ALWAYS_LOG(level, ...) thorin::Log::log((level), __VA_ARGS__)
#ifndef NDEBUG
#define MAYBE_LOG(level, ...) ALWAYS_LOG(level, __VA_ARGS__)
#else
#define MAYBE_LOG(level, ...) do {} while (false)
#endif

#define ELOG(def, ...) thorin::Log::error((def)->location(), __VA_ARGS__)
#define ELOG_LOC(loc, ...) thorin::Log::error(loc, __VA_ARGS__)
#define WLOG(def, ...) ALWAYS_LOG(thorin::Log::Warn, (def)->location(), __VA_ARGS__)
#define ILOG(def, ...) ALWAYS_LOG(thorin::Log::Info, (def)->location(), __VA_ARGS__)
#define VLOG(...) MAYBE_LOG(thorin::Log::Verbose, Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#define DLOG(...) MAYBE_LOG(thorin::Log::Debug,   Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#define VLOG_SCOPE(s) { \
    VLOG("*** BEGIN: " #s " {{"); \
    (s); \
    VLOG("}}"); \
}

#endif
