#ifndef THORIN_MATCH_H
#define THORIN_MATCH_H

#include "thorin/def.h"

namespace thorin {

template <unsigned Min, unsigned Max = Min>
struct MatchTag {
    static bool match(const Def* def) { return Min <= def->tag() && def->tag() <= Max; }
};

template <size_t Op, typename Matcher>
struct MatchOp {
    static bool match(const Def* def) { return Matcher::match(def->op(Op)); }
};

template <typename Matcher>
struct MatchType {
    static bool match(const Def* def) { return Matcher::match(def->type()); }
};

template <typename Matcher, typename... Args>
struct MatchAnd {
    static bool match(const Def* def) { return Matcher::match(def) && MatchAnd<Args...>::match(def); }
};

template <typename Matcher>
struct MatchAnd<Matcher> {
    static bool match(const Def* def) { return Matcher::match(def); }
};

template <typename Matcher, typename... Args>
struct MatchOr {
    static bool match(const Def* def) { return Matcher::match(def) || MatchOr<Args...>::match(def); }
};

template <typename Matcher>
struct MatchOr<Matcher> {
    static bool match(const Def* def) { return Matcher::match(def); }
};

struct IsLiteral {
    static bool match(const Def* def) { return def->isa<Lit>(); }
};

using IsKind  = MatchType<MatchTag<Tag::Universe>>;
using IsType  = MatchType<MatchTag<Tag::KindStar>>;
using IsValue = MatchType<IsType>;

}

#endif
