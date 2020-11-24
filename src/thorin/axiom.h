#ifndef THORIN_AXIOM_H
#define THORIN_AXIOM_H

#include "thorin/lam.h"

namespace thorin {

class Axiom : public Def {
private:
    Axiom(NormalizeFn normalizer, const Def* type, tag_t tag, flags_t flags, const Def* dbg);

public:
    /// @name misc getters
    //@{
    tag_t tag() const { return fields() >> 32_u64; }
    flags_t flags() const { return fields(); }
    NormalizeFn normalizer() const { return normalizer_depth_.ptr(); }
    u16 currying_depth() const { return normalizer_depth_.index(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Axiom;
    friend class World;
};

template<class T, class U> bool has(T flags, U option) { return (flags & option) == option; }

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

template<tag_t tag> struct Tag2Def_ { using type = App; };
template<> struct Tag2Def_<Tag::Mem> { using type = Axiom; };
template<tag_t tag> using Tag2Def = typename Tag2Def_<tag>::type;

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

template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(               const Def* d) { assert( isa<t>(   d) ); return {std::get<0>(get_axiom(d)), d->as<App>()}; }
template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(Tag2Enum<t> f, const Def* d) { assert((isa<t>(f, d))); return {std::get<0>(get_axiom(d)), d->as<App>()}; }

/// Checks whether @p type is an @p Int or a @p Real and returns its bound or width, respectively.
inline const Def* isa_sized_type(const Def* type) {
    if (auto int_ = isa<Tag:: Int>(type)) return int_->arg();
    if (auto real = isa<Tag::Real>(type)) return real->arg();
    return nullptr;
}

constexpr uint64_t width2bound(uint64_t n) {
    assert(n != 0);
    return n == 64 ? 0 : (1_u64 << n);
}

constexpr std::optional<uint64_t> bound2width(uint64_t n) {
    if (n == 0) return 64;
    if (is_power_of_2(n)) return log2(n);
    return {};
}

bool is_memop(const Def* def);

}

#endif
