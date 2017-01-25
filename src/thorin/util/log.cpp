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

bool Log::print_loc_ = true;
Log::Level Log::min_level_ = Log::Info;
std::ostream* Log::stream_ = nullptr;

std::ostream& Log::stream() { return *stream_; }
void Log::set(Level min_level, std::ostream* stream, bool print_loc) { set_min_level(min_level); set_stream(stream); set_print_loc(print_loc); }
Log::Level Log::min_level() { return min_level_; }
void Log::set_stream(std::ostream* stream) { stream_ = stream; }
void Log::set_min_level(Log::Level min_level) { min_level_ = min_level; }
void Log::set_print_loc(bool print_loc) { print_loc_ = print_loc; }

std::ostream* Log::get_stream() { return stream_; }
Log::Level Log::get_min_level() { return min_level_; }
bool Log::get_print_loc() { return print_loc_; }

std::string Log::level2string(Level level) {
    switch (level) {
        case Error: return "E";
        case Warn:  return "W";
        case Info:  return "I";
        case Debug: return "D";
    }
    THORIN_UNREACHABLE;
}

int Log::level2color(Level level) {
    switch (level) {
        case Error: return 1;
        case Warn:  return 3;
        case Info:  return 2;
        case Debug: return 4;
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
