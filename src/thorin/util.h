#ifndef THORIN_UTIL_H
#define THORIN_UTIL_H

#include "thorin/def.h"

// TODO clean up this mess

namespace thorin {

bool is_unit(const Def*);
bool is_const(const Def*);
bool is_tuple_arg_of_app(const Def*);
bool is_memop(const Def* def);

Array<const Def*> merge(const Def* def, Defs defs);
const Def* merge_sigma(const Def* def, Defs defs);
const Def* merge_tuple(const Def* def, Defs defs);

std::string tuple2str(const Def*);

bool visit_uses(Lam* lam, std::function<bool(Lam*)> func, bool include_globals = true);
bool visit_capturing_intrinsics(Lam* lam, std::function<bool(Lam*)> func, bool include_globals);

inline bool is_passed_to_accelerator(Lam* lam, bool include_globals = true) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->is_accelerator(); }, include_globals);
}

inline bool is_passed_to_intrinsic(Lam* lam, Lam::Intrinsic intrinsic, bool include_globals = true) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->intrinsic() == intrinsic; }, include_globals);
}

std::tuple<const Axiom*, u16> get_axiom(const Def*);

// TODO put this somewhere else
template<tag_t tag> struct Tag2Def_   { using type = App; };
template<> struct Tag2Def_<Tag::End> { using type = Axiom; };
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

inline std::optional<nat_t> get_width(const Def* type) {
    if (false) {}
    else if (auto int_ = isa<Tag::Int >(type)) return isa_lit<nat_t>(int_->arg());
    else if (auto real = isa<Tag::Real>(type)) return isa_lit<nat_t>(real->arg());
    return {};
}

template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(               const Def* d) { assert( isa<t>(   d) ); return {std::get<0>(get_axiom(d)), d->as<App>()}; }
template<tag_t t> Query<Tag2Enum<t>, Tag2Def<t>> as(Tag2Enum<t> f, const Def* d) { assert((isa<t>(f, d))); return {std::get<0>(get_axiom(d)), d->as<App>()}; }

void app_to_dropped_app(Lam* src, Lam* dst, const App* app);

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
