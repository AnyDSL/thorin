#ifndef THORIN_NORMALIZE_H
#define THORIN_NORMALIZE_H

namespace thorin {

class Def;

const Def* normalize_bit    (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_bitcast(const Def*, const Def*, const Def*, const Def*);
const Def* normalize_lea    (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_sizeof (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_load   (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_store  (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_tangent(const Def*, const Def*, const Def*, const Def*);

template<Bit > const Def* normalize_Bit (const Def*, const Def*, const Def*, const Def*);
template<Shr > const Def* normalize_Shr (const Def*, const Def*, const Def*, const Def*);
template<WOp > const Def* normalize_WOp (const Def*, const Def*, const Def*, const Def*);
template<ZOp > const Def* normalize_ZOp (const Def*, const Def*, const Def*, const Def*);
template<ROp > const Def* normalize_ROp (const Def*, const Def*, const Def*, const Def*);
template<ICmp> const Def* normalize_ICmp(const Def*, const Def*, const Def*, const Def*);
template<RCmp> const Def* normalize_RCmp(const Def*, const Def*, const Def*, const Def*);
template<Conv> const Def* normalize_Conv(const Def*, const Def*, const Def*, const Def*);
template<PE  > const Def* normalize_PE  (const Def*, const Def*, const Def*, const Def*);

}

#endif
