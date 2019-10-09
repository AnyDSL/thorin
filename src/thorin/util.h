#ifndef THORIN_UTIL_H
#define THORIN_UTIL_H

#include "thorin/def.h"

// TODO clean up this mess

namespace thorin {

bool is_unit(const Def*);
bool is_const(const Def*);
bool is_tuple_arg_of_app(const Def*);
bool is_memop(const Def* def);
bool is_symmetric(const Def* def);


template<class T, class U> bool has(T flags, U option) { return (flags & option) == option; }

Array<const Def*> merge(const Def* def, Defs defs);
const Def* merge_sigma(const Def* def, Defs defs);
const Def* merge_tuple(const Def* def, Defs defs);

const Def* proj(const Def* def, u64 i);

std::string tuple2str(const Def*);
std::tuple<const Axiom*, u16> get_axiom(const Def*);

// TODO put this somewhere else
template<tag_t tag> struct Tag2Def_   { using type = App; };
template<tag_t tag> using Tag2Def = typename Tag2Def_<tag>::type;

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

inline const Def* get_width_as_def(const Def* type) {
    if (false) {}
    else if (auto int_ = isa<Tag:: Int>(type)) return int_->arg();
    else if (auto sint = isa<Tag::SInt>(type)) return sint->arg();
    else if (auto real = isa<Tag::Real>(type)) return real->arg();
    return nullptr;
}

inline std::optional<nat_t> get_width(const Def* type) {
    if (auto def = get_width_as_def(type))
        return isa_lit<nat_t>(def);
    return std::nullopt;
}

inline const Def* infer_width(const Def* def) { return get_width_as_def(def->type()); }

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
