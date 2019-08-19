#ifndef THORIN_UTIL_H
#define THORIN_UTIL_H

#include "thorin/def.h"

// TODO clean up this mess

namespace thorin {

bool is_unit(const Def*);
bool is_const(const Def*);
bool is_primlit(const Def*, int64_t);
bool is_minus_zero(const Def*);

bool is_tuple_arg_of_app(const Def*);

inline bool is_primtype (const Def* t) { return t->isa<PrimType>() && thorin::is_primtype(t->flags()); }
inline bool is_type_ps  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_ps (t->flags()); }
inline bool is_type_pu  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_pu (t->flags()); }
inline bool is_type_qs  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_qs (t->flags()); }
inline bool is_type_qu  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_qu (t->flags()); }
inline bool is_type_pf  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_pf (t->flags()); }
inline bool is_type_qf  (const Def* t) { return t->isa<PrimType>() && thorin::is_type_qf (t->flags()); }
inline bool is_type_p   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_p  (t->flags()); }
inline bool is_type_q   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_q  (t->flags()); }
inline bool is_type_s   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_s  (t->flags()); }
inline bool is_type_u   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_u  (t->flags()); }
inline bool is_type_i   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_i  (t->flags()); }
inline bool is_type_f   (const Def* t) { return t->isa<PrimType>() && thorin::is_type_f  (t->flags()); }
inline bool is_type_bool(const Def* t) { return t->isa<PrimType>() && t->flags() == Node_PrimType_bool; }

inline bool is_arity(const Def* def) { return def->type()->isa<KindArity>(); }
inline bool is_mem        (const Def* def) { return def->type()->isa<MemType>(); }
inline bool is_zero       (const Def* def) { return is_primlit(def, 0); }
inline bool is_one        (const Def* def) { return is_primlit(def, 1); }
inline bool is_allset     (const Def* def) { return is_primlit(def, -1); }
bool is_not        (const Def* def);
bool is_minus      (const Def* def);
bool is_div_or_rem (const Def* def);
bool is_commutative(const Def* def);
bool is_associative(const Def* def);

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
