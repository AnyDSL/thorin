#ifndef THORIN_UTIL_BOX_H
#define THORIN_UTIL_BOX_H

#include <cstring>

#include "thorin/util/assert.h"
#include "thorin/util/types.h"

namespace thorin {

union Box {
public:
    Box()      { reset(); }
    Box( u1 u) { reset();   u1_ = u.get(); }
    Box( u8 u) { reset();   u8_ = u; }
    Box(u16 u) { reset();  u16_ = u; }
    Box(u32 u) { reset();  u32_ = u; }
    Box(u64 u) { reset();  u64_ = u; }
    Box(f32 f) { reset();  f32_ = f; }
    Box(f64 f) { reset();  f64_ = f; }
    Box(bool b){ reset();   u1_ = b; }

    bool operator == (const Box& other) const { return bcast<uint64_t, Box>(*this) == bcast<uint64_t, Box>(other); }
    // see blow for specializations
    template <typename T> inline T get() { THORIN_UNREACHABLE; } 
     u1  get_u1() const { return u1(u1_); }
     u8  get_u8() const { return u8_; }
    u16 get_u16() const { return u16_; }
    u32 get_u32() const { return u32_; }
    u64 get_u64() const { return u64_; }
    f32 get_f32() const { return f32_; }
    f64 get_f64() const { return f64_; }

private:
    void reset() { memset(this, 0, sizeof(Box)); }

    bool  u1_;
    u8    u8_;
    u16  u16_;
    u32  u32_;
    u64  u64_;
    f32  f32_;
    f64  f64_;
};

template <> inline  u1 Box::get< u1>() { return  u1_; }
template <> inline  u8 Box::get< u8>() { return  u8_; }
template <> inline u16 Box::get<u16>() { return u16_; }
template <> inline u32 Box::get<u32>() { return u32_; }
template <> inline u64 Box::get<u64>() { return u64_; }
template <> inline f32 Box::get<f32>() { return f32_; }
template <> inline f64 Box::get<f64>() { return f64_; }

inline size_t hash_value(Box box) { return hash_value(bcast<u64, Box>(box)); }

}

#endif
