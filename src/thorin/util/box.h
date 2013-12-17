#ifndef THORIN_UTIL_BOX_H
#define THORIN_UTIL_BOX_H

#include <cstring>

#include "thorin/util/assert.h"
#include "thorin/util/types.h"

namespace thorin {

union Box {
public:
    Box()      { reset(); }
//#define THORIN_ALL_TYPE(T) Box(T val) { reset(); T##_ = val; }
//#include "thorin/tables/primtypetable.h"
    Box( s1 u) { reset();   ps1_ = u.get(); }
    Box( s8 u) { reset();   ps8_ = u; }
    Box(s16 u) { reset();  ps16_ = u; }
    Box(s32 u) { reset();  ps32_ = u; }
    Box(s64 u) { reset();  ps64_ = u; }
    //Box( u1 u) { reset();   u1_ = u.get(); }
    Box( u8 u) { reset();   pu8_ = u; }
    Box(u16 u) { reset();  pu16_ = u; }
    Box(u32 u) { reset();  pu32_ = u; }
    Box(u64 u) { reset();  pu64_ = u; }
    Box(f32 f) { reset();  pf32_ = f; }
    Box(f64 f) { reset();  pf64_ = f; }
    Box(bool b){ reset();   pu1_ = b; }

    bool operator == (const Box& other) const { return bcast<uint64_t, Box>(*this) == bcast<uint64_t, Box>(other); }
    // see blow for specializations
    template <typename T> inline T get() { THORIN_UNREACHABLE; } 
#define THORIN_ALL_TYPE(T) \
    T get_##T() const { return T##_; }
#include "thorin/tables/primtypetable.h"
     s1 get_s1()  const { return ps1_; }
     s8 get_s8()  const { return ps8_; }
    s16 get_s16() const { return ps16_; }
    s32 get_s32() const { return ps32_; }
    s64 get_s64() const { return ps64_; }
     u1 get_u1()  const { return pu1_; }
     u8 get_u8()  const { return pu8_; }
    u16 get_u16() const { return pu16_; }
    u32 get_u32() const { return pu32_; }
    u64 get_u64() const { return pu64_; }
    f32 get_f32() const { return pf32_; }
    f64 get_f64() const { return pf64_; }

private:
    void reset() { memset(this, 0, sizeof(Box)); }

#define THORIN_ALL_TYPE(T) T T##_;
#include "thorin/tables/primtypetable.h"
};

static_assert(sizeof(Box) == sizeof(uint64_t), "Box has incorrect size in bytes");

template <> inline  s1 Box::get< s1>() { return  ps1_; }
template <> inline  s8 Box::get< s8>() { return  ps8_; }
template <> inline s16 Box::get<s16>() { return ps16_; }
template <> inline s32 Box::get<s32>() { return ps32_; }
//template <> inline  u1 Box::get< u1>() { return  u1_; }
template <> inline  u8 Box::get< u8>() { return  pu8_; }
template <> inline u16 Box::get<u16>() { return pu16_; }
template <> inline u32 Box::get<u32>() { return pu32_; }
template <> inline u64 Box::get<u64>() { return pu64_; }
template <> inline f32 Box::get<f32>() { return pf32_; }
template <> inline f64 Box::get<f64>() { return pf64_; }

inline size_t hash_value(Box box) { return hash_value(bcast<u64, Box>(box)); }

}

#endif
