#include "thorin/error.h"

#include "thorin/def.h"

#include "thorin/util/stream.h"

namespace thorin {

template<class... Args> void err(const char* fmt, Args&&... args) { errf(fmt, std::forward<Args&&>(args)...); std::abort(); }

void ErrorHandler::index_out_of_range(const Def* arity, const Def* index) {
    err("index '{}' does not fit within arity '{}'", index, arity);
}

void ErrorHandler::ill_typed_app(const Def* callee, const Def* arg) {
    err("cannot pass argument '{} of type '{}' to '{}' of domain '{}'", arg, arg->type(), callee, callee->type()->as<Pi>()->domain());
}

void ErrorHandler::incomplete_match(const Match* match) {
    err("match {} is incomplete", match);
}

void ErrorHandler::redundant_match_case(const Match* match, const Ptrn* first) {
    err("match {} has a redundant case: {}", match, first);
}

}
