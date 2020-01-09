#include "thorin/util/log.h"
#include "thorin/util/utility.h"

// For colored output
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace thorin {

Log::Level Log::min_level_ = Log::Error;
std::ostream* Log::stream_ = nullptr;

std::ostream& Log::stream() { return *stream_; }
Log::Level Log::min_level() { return min_level_; }
void Log::set_stream(std::ostream* stream) { stream_ = stream; }
void Log::set_min_level(Log::Level min_level) { min_level_ = min_level; }
std::ostream* Log::get_stream() { return stream_; }
Log::Level Log::get_min_level() { return min_level_; }

void Log::set(Level min_level, std::ostream* stream) {
    set_min_level(min_level);
    set_stream(stream);
}

std::string Log::level2string(Level level) {
    switch (level) {
#ifdef _MSC_VER
        case Error:   return "error";
        case Warn:    return "warning";
        case Info:    return "info";
        case Verbose: return "verbose";
        case Debug:   return "debug";
#else
        case Error:   return "E";
        case Warn:    return "W";
        case Info:    return "I";
        case Verbose: return "V";
        case Debug:   return "D";
#endif
    }
    THORIN_UNREACHABLE;
}

int Log::level2color(Level level) {
    switch (level) {
        case Error:   return 1;
        case Warn:    return 3;
        case Info:    return 2;
        case Verbose: return 4;
        case Debug:   return 4;
    }
    THORIN_UNREACHABLE;
}

#ifdef COLORIZE_LOG
std::string Log::colorize(const std::string& str, int color) {
    if (isatty(fileno(stdout))) {
        const char c = '0' + color;
        return "\033[1;3" + (c + ('m' + str)) + "\033[0m";
    }
#else
std::string Log::colorize(const std::string& str, int) {
#endif
    return str;
}


}
