#ifndef THORIN_UTIL_H
#define THORIN_UTIL_H

#include "thorin/def.h"

// TODO clean up this mess

namespace thorin {

constexpr uint64_t width2bound(uint64_t n) {
    assert(n != 0);
    return n == 64 ? 0 : (1_u64 << n);
}

constexpr std::optional<uint64_t> bound2width(uint64_t n) {
    if (n == 0) return 64;
    if (is_power_of_2(n)) return log2(n);
    return {};
}

bool is_unit(const Def*);
bool is_tuple_arg_of_app(const Def*);
bool is_memop(const Def* def);
bool is_symmetric(const Def* def);

template<class T, class U> bool has(T flags, U option) { return (flags & option) == option; }

Array<const Def*> merge(const Def* def, Defs defs);
const Def* merge_sigma(const Def* def, Defs defs);
const Def* merge_tuple(const Def* def, Defs defs);

std::string tuple2str(const Def*);
std::tuple<const Axiom*, u16> get_axiom(const Def*);

template<class F, class D>
class Query {
public:
    Query()
        : axiom_(nullptr)
        , def_(nullptr)
    {}
    Query(const Axiom* axiom, const D* def)
        : axiom_(axiom)
        , def_(def)
    {}

    const Axiom* axiom() const { return axiom_; }
    tag_t tag() const { return axiom()->tag(); }
    F flags() const { return F(axiom()->flags()); }
    void clear() { axiom_ = nullptr; def_ = nullptr; }

    const D* operator->() const { return def_; }
    operator const D*() const { return def_; }
    explicit operator bool() { return axiom_ != nullptr; }

private:
    const Axiom* axiom_;
    const D* def_;
};

template<tag_t tag>
Query<Tag2Enum<tag>, Tag2Def<tag>> isa(const Def* def) {
    auto [axiom, currying_depth] = get_axiom(def);
    if (axiom && axiom->tag() == tag && currying_depth == 0)
        return {axiom, def->as<Tag2Def<tag>>()};
    return {};
}

template<tag_t tag>
Query<Tag2Enum<tag>, Tag2Def<tag>> isa(Tag2Enum<tag> flags, const Def* def) {
    auto [axiom, currying_depth] = get_axiom(def);
    if (axiom && axiom->tag() == tag && axiom->flags() == flags_t(flags) && currying_depth == 0)
        return {axiom, def->as<Tag2Def<tag>>()};
    return {};
}

/// Checks whether @p type is an @p Int or a @p Real and returns its bound or width, respectively.
inline const Def* isa_sized_type(const Def* type) {
    if (auto int_ = isa<Tag:: Int>(type)) return int_->arg();
    if (auto real = isa<Tag::Real>(type)) return real->arg();
    return nullptr;
}

inline const Def* infer_size(const Def* def) { return isa_sized_type(def->type()); }

template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(               const Def* d) { assert( isa<t>(   d) ); return {std::get<0>(get_axiom(d)), d->as<App>()}; }
template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(Tag2Enum<t> f, const Def* d) { assert((isa<t>(f, d))); return {std::get<0>(get_axiom(d)), d->as<App>()}; }

class Peek {
public:
    Peek() {}
    Peek(const Def* def, Lam* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lam* from() const { return from_; }

private:
    const Def* def_;
    Lam* from_;
};

size_t get_param_index(const Def* def);
Lam* get_param_lam(const Def* def);
std::vector<Peek> peek(const Def*);

}

#endif
