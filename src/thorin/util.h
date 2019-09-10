#ifndef THORIN_UTIL_H
#define THORIN_UTIL_H

#include "thorin/def.h"

// TODO clean up this mess

namespace thorin {

bool is_unit(const Def*);
bool is_const(const Def*);
bool is_tuple_arg_of_app(const Def*);

inline bool is_arity(const Def* def) { return def->type()->isa<KindArity>(); }
inline bool is_memop      (const Def* def) { return def->num_ops() >= 1 && def->op(0)->type()->isa<Mem>(); }

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

template<class F>
class Query {
public:
    Query()
        : axiom_(nullptr)
        , app_(nullptr)
    {}
    Query(const Axiom* axiom, const App* app)
        : axiom_(axiom)
        , app_(app)
    {}

    const Axiom* axiom() const { return axiom_; }
    tag_t tag() const { return axiom()->tag(); }
    F flags() const { return F(axiom()->flags()); }

    const App* operator->() const { return app_; }
    operator const App*() const { return app_; }
    explicit operator bool() { return axiom_ != nullptr; }

private:
    const Axiom* axiom_;
    const App* app_;
};

template<tag_t tag>
Query<Tag2Enum<tag>> isa(const Def* def) {
    auto [axiom, currying_depth] = get_axiom(def);
    if (axiom->tag() == tag && currying_depth == 0)
        return {axiom, def->as<App>()};
    return {};
}

template<tag_t tag, Tag2Enum<tag> flags>
Query<Tag2Enum<tag>> isa(const Def* def) {
    auto [axiom, currying_depth] = get_axiom(def);
    if (axiom->tag() == tag && axiom->flags() == flags_t(flags) && currying_depth == 0)
        return {axiom, def->as<App>()};
    return {};
}

template<tag_t t>                Query<Tag2Enum<t>> as(const Def* d) { assert( isa<t   >(d) ); return {std::get<0>(get_axiom(d)), d->as<App>()}; }
template<tag_t t, Tag2Enum<t> f> Query<Tag2Enum<t>> as(const Def* d) { assert((isa<t, f>(d))); return {std::get<0>(get_axiom(d)), d->as<App>()}; }

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
