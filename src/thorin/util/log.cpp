#include "log.h"

namespace thorin {

Log::Level Log::max_level_ = Log::Info;
std::ostream* Log::stream_ = nullptr;

}
