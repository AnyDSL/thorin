#ifndef THORIN_NORMALIZE_H
#define THORIN_NORMALIZE_H

namespace thorin {

const Def* normalize_select(const Def*, const Def*, const Def*, const Def*);
const Def* normalize_sizeof(const Def*, const Def*, const Def*, const Def*);

template<WOp > const Def* normalize_WOp (const Def*, const Def*, const Def*, const Def*);
template<ZOp > const Def* normalize_ZOp (const Def*, const Def*, const Def*, const Def*);
template<IOp > const Def* normalize_IOp (const Def*, const Def*, const Def*, const Def*);
template<ROp > const Def* normalize_ROp (const Def*, const Def*, const Def*, const Def*);
template<ICmp> const Def* normalize_ICmp(const Def*, const Def*, const Def*, const Def*);
template<RCmp> const Def* normalize_RCmp(const Def*, const Def*, const Def*, const Def*);
template<Cast> const Def* normalize_Cast(const Def*, const Def*, const Def*, const Def*);

#define CODE(T, o) normalize_ ## T<T::o>,
constexpr std::array<Def::NormalizeFn, Num<WOp>>  normalizers_WOp  = { THORIN_W_OP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<ZOp>>  normalizers_ZOp  = { THORIN_Z_OP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<IOp>>  normalizers_IOp  = { THORIN_I_OP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<ROp>>  normalizers_ROp  = { THORIN_R_OP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<ICmp>> normalizers_ICmp = { THORIN_I_CMP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<RCmp>> normalizers_RCmp = { THORIN_R_CMP(CODE) };
constexpr std::array<Def::NormalizeFn, Num<Cast>> normalizers_Cast = { THORIN_CAST(CODE) };
#undef CODE

}

#endif
