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
    static void set(Level min_level, std::ostream* stream);
    static Level min_level();
    static void set_stream(std::ostream* stream);
    static void set_min_level(Level min_level);
    static std::string level2string(Level);
    static int level2color(Level);
    static std::string colorize(const std::string&, int);

    template<typename... Args>
    static void log(Level level, Location location, const char* fmt, Args... args) {
        if (Log::get_stream() && Log::get_min_level() <= level) {
            std::ostringstream oss;
            oss << location;
            #ifdef _MSC_VER
            streamf(Log::stream(), "{}: {}: ", colorize(oss.str(), 7), colorize(level2string(level), level2color(level)));
            #else
            streamf(Log::stream(), "{}:{}: ", colorize(level2string(level), level2color(level)), colorize(oss.str(), 7));
            #endif
            streamf(Log::stream(), fmt, std::forward<Args>(args)...) << std::endl;
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

    static std::ostream* stream_;
    static Level min_level_;
};

template<typename... Args> std::ostream& outf(const char* fmt, Args... args) { return streamf(std::cout, fmt, std::forward<Args>(args)...); }
template<typename... Args> std::ostream& errf(const char* fmt, Args... args) { return streamf(std::cerr, fmt, std::forward<Args>(args)...); }

// TODO don't use macros
// TODO remove static state from these things
#define EDEF(def, ...) thorin::Log::error(                 (def)->location(), __VA_ARGS__)
#define WDEF(def, ...) thorin::Log::log(thorin::Log::Warn, (def)->location(), __VA_ARGS__)
#define IDEF(def, ...) thorin::Log::log(thorin::Log::Info, (def)->location(), __VA_ARGS__)

#define ELOG(...) thorin::Log::log(thorin::Log::Error,   Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#define WLOG(...) thorin::Log::log(thorin::Log::Warn,    Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#define ILOG(...) thorin::Log::log(thorin::Log::Info,    Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#define VLOG(...) thorin::Log::log(thorin::Log::Verbose, Location(__FILE__, __LINE__, -1), __VA_ARGS__)

#ifndef NDEBUG
#define DLOG(...) thorin::Log::log(thorin::Log::Debug,   Location(__FILE__, __LINE__, -1), __VA_ARGS__)
#else
#define DLOG(...) do {} while (false)
#endif

}

#endif
