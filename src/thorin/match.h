#ifndef THORIN_MATCH_H
#define THORIN_MATCH_H

#include "thorin/def.h"

namespace thorin {

template <unsigned Min, unsigned Max>
struct MatchTag {
    static bool match(const Def* def) {
        return def->tag() >= Min && def->tag() <= Max;
    }
};

template <unsigned Tag>
using MatchTagExact = MatchTag<Tag, Tag>;

template <size_t Op, typename Matcher>
struct MatchOp {
    static bool match(const Def* def) {
        return Matcher::match(def->op(Op));
    }
};

template <typename Matcher>
struct MatchType {
    static bool match(const Def* def) {
        return Matcher::match(def->type());
    }
};

template <typename Matcher, typename... Args>
struct MatchManyAnd {
    static bool match(const Def* def) {
        return Matcher::match(def) && MatchManyAnd<Args...>::match(def);
    }
};

template <typename Matcher>
struct MatchManyAnd<Matcher> {
    static bool match(const Def* def) {
        return Matcher::match(def);
    }
};

template <typename Left, typename Right>
using MatchAnd = MatchManyAnd<Left, Right>;

template <typename Matcher, typename... Args>
struct MatchManyOr {
    static bool match(const Def* def) {
        return Matcher::match(def) || MatchManyOr<Args...>::match(def);
    }
};

template <typename Matcher>
struct MatchManyOr<Matcher> {
    static bool match(const Def* def) {
        return Matcher::match(def);
    }
};

template <typename Left, typename Right>
using MatchOr = MatchManyOr<Left, Right>;

struct IsLiteral {
    static bool match(const Def* def) {
        return def->isa<Lit>();
    }
};

using IsKind  = MatchType<MatchTagExact<Node_Universe>>;
using IsType  = MatchType<MatchTagExact<Node_KindStar>>;
using IsValue = MatchType<IsType>;

}

#endif
