#include "thorin/error.h"

#include "thorin/util/stream.h"

namespace thorin {

template<class... Args> void err(const char* fmt, Args&&... args) { errf(fmt, std::forward<Args&&>(args)...); std::abort(); }

void ErrorHandler::index_out_of_range(uint64_t arity, uint64_t index) {
    err("index literal '{}' does not fit within arity '{}'", index, arity);
}

void ErrorHandler::incomplete_match(const Match* match) {
    err("match {} is incomplete", match);
}

void ErrorHandler::redundant_match_cases(const Match* match, size_t index) {
    err("match {} has redundant cases starting from index {}", match, index + 1);
}

}
