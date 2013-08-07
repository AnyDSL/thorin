#ifndef ANYDSL2_DSLU_BOX_H
#define ANYDSL2_DSLU_BOX_H

#include <cstring>

#include "anydsl2/util/assert.h"
#include "anydsl2/util/types.h"

namespace anydsl2 {

typedef const char* const_char_ptr;

union Box {
public:

    Box()      { reset(); }
    Box( u1 u) { reset(); bool_ = u.get(); }
    Box( u8 u) { reset();   u8_ = u; }
    Box(u16 u) { reset();  u16_ = u; }
    Box(u32 u) { reset();  u32_ = u; }
    Box(u64 u) { reset();  u64_ = u; }
    Box(f32 f) { reset();  f32_ = f; }
    Box(f64 f) { reset();  f64_ = f; }

    Box(bool b)  { reset(); bool_ = b; }
    Box(char c)  { reset(); char_ = c; }
    Box(void* p) { reset(); ptr_  = p; }
    Box(const_char_ptr s) { reset(); const_char_ptr_ = s; }

    bool operator == (const Box& other) const {
        return bcast<uint64_t, Box>(*this) == bcast<uint64_t, Box>(other);
    }

    template <typename T>
    inline T get() { ANYDSL2_UNREACHABLE; }

     u1  get_u1() const { return u1(bool_); }
     u8  get_u8() const { return u8_; }
    u16 get_u16() const { return u16_; }
    u32 get_u32() const { return u32_; }
    u64 get_u64() const { return u64_; }
    f32 get_f32() const { return f32_; }
    f64 get_f64() const { return f64_; }
    void* get_ptr() const { return ptr_; }

private:

    void reset() { memset(this, 0, sizeof(Box)); }

    u8    u8_;
    u16  u16_;
    u32  u32_;
    u64  u64_;
    f32  f32_;
    f64  f64_;

    bool bool_;
    char  char_;
    void* ptr_;
    const_char_ptr const_char_ptr_;

};

template <> inline  u1 Box::get< u1>() { return  u1(bool_); }
template <> inline  u8 Box::get< u8>() { return  u8_; }
template <> inline u16 Box::get<u16>() { return u16_; }
template <> inline u32 Box::get<u32>() { return u32_; }
template <> inline u64 Box::get<u64>() { return u64_; }
template <> inline f32 Box::get<f32>() { return f32_; }
template <> inline f64 Box::get<f64>() { return f64_; }
template <> inline bool Box::get<bool>() { return bool_; }
template <> inline const_char_ptr Box::get<const_char_ptr>() { return const_char_ptr_; }
template <> inline char Box::get<char>() { return char_; }
template <> inline void* Box::get<void*>() { return ptr_; }

inline size_t hash_value(Box box) { return hash_value(bcast<u64, Box>(box)); }

} // namespace anydsl2

#endif // ANYDSL2_DSLU_BOX_H
