#ifndef THORIN_NORMALIZE_H
#define THORIN_NORMALIZE_H

namespace thorin {

const Def* normalize_r2r   (const Def*, const Def*, const Def*, const Def*);
const Def* normalize_select(const Def*, const Def*, const Def*, const Def*);
const Def* normalize_sizeof(const Def*, const Def*, const Def*, const Def*);

template<WOp > const Def* normalize_WOp (const Def*, const Def*, const Def*, const Def*);
template<ZOp > const Def* normalize_ZOp (const Def*, const Def*, const Def*, const Def*);
template<IOp > const Def* normalize_IOp (const Def*, const Def*, const Def*, const Def*);
template<ROp > const Def* normalize_ROp (const Def*, const Def*, const Def*, const Def*);
template<ICmp> const Def* normalize_ICmp(const Def*, const Def*, const Def*, const Def*);
template<RCmp> const Def* normalize_RCmp(const Def*, const Def*, const Def*, const Def*);
template<Conv> const Def* normalize_Conv(const Def*, const Def*, const Def*, const Def*);

}

#endif
