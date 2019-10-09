#include "thorin/error.h"

#include "thorin/util/stream.h"

namespace thorin {

template<class... Args> void err(const char* fmt, Args&&... args) { errf(fmt, std::forward<Args&&>(args)...); std::abort(); }

void DefaultHandler::index_out_of_range(uint64_t arity, uint64_t index) {
    err("index literal '{}' does not fit within arity '{}'", index, arity);
}

void DefaultHandler::empty_cases() {
    err("match must take at least one case");
}

void DefaultHandler::match_cases_inconsistent(const Def* t1, const Def* t2) {
    err("cases' types are inconsistent with each other, got {} and {}", t1, t2);
}

}
