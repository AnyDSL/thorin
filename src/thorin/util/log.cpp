#include "thorin/util/log.h"

#include "thorin/util/assert.h"

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

char Log::level2char(Level level) {
    switch (level) {
        case Error: return 'E';
        case Warn:  return 'W';
        case Info:  return 'I';
        case Debug: return 'D';
    }
    THORIN_UNREACHABLE;
}

}
