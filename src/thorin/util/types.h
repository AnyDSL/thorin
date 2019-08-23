#ifndef THORIN_UTIL_TYPES_H
#define THORIN_UTIL_TYPES_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif
#define HALF_ROUND_STYLE 1
#define HALF_ROUND_TIES_TO_EVEN 1
#include <half.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "thorin/util/cast.h"
#include "thorin/util/hash.h"

namespace thorin {

using half_float::half;

// This code assumes two-complement arithmetic for unsigned operations.
// This is *implementation-defined* but *NOT* *undefined behavior*.

struct BottomException {};

template<class ST, bool wrap>
class SInt {
public:
    typedef typename std::make_unsigned<ST>::type UT;
    static_assert(std::is_signed<ST>::value, "ST must be a signed type");

    SInt() = default;
    SInt(ST data)
        : data_(data)
    {}

    SInt operator-() const {
        if (data_ == std::numeric_limits<ST>::min()) {
            if (!wrap)
                throw BottomException();
            else
                return SInt(0);
        }
        return SInt(-data_);
    }

    SInt operator+(SInt other) const {
        SInt res(UT(this->data_) + UT(other.data_));
        if (!wrap && (this->is_neg() == other.is_neg()) && (this->is_neg() != res.is_neg()))
            throw BottomException();
        return res;
    }

    SInt operator-(SInt other) const { return *this + SInt(other.minus()); }

    SInt operator*(SInt other) const {
        ST a = this->data_, b = other.data_;

        if (!wrap) {
            if (a > ST(0)) {
                if (b > ST(0)) {
                    if (a > (std::numeric_limits<ST>::max() / b))
                        throw BottomException();
                } else {
                    if (b < (std::numeric_limits<ST>::min() / a))
                        throw BottomException();
                }
            } else {
                if (b > ST(0)) {
                    if (a < (std::numeric_limits<ST>::min() / b))
                        throw BottomException();
                } else {
                    if ( (a != ST(0)) && (b < (std::numeric_limits<ST>::max() / a)))
                        throw BottomException();
                }
            }
        }

        return UT(a) * UT(b);
    }

    void div_check(SInt other) const {
        if (other.data_ == ST(0) || (this->data_ == std::numeric_limits<ST>::min() && other.data_ == ST(-1)))
            throw BottomException();
    }

    SInt operator/(SInt other) const { div_check(other); return SInt(this->data_ / other.data_); }
    SInt operator%(SInt other) const { div_check(other); return SInt(this->data_ % other.data_); }

    SInt operator<<(SInt other) const {
        if (other.data_ >= std::numeric_limits<ST>::digits+1 || other.is_neg())
            throw BottomException();

        // TODO: actually this is correct, but see: https://github.com/AnyDSL/impala/issues/34
        //if (!wrap && (this->is_neg() || this->data_ > std::numeric_limits<ST>::max() >> other.data_))
        //    throw BottomException();

        return ST(UT(this->data_) << UT(other.data_));
    }

    SInt operator& (SInt other) const { return this->data_ & other.data_; }
    SInt operator| (SInt other) const { return this->data_ | other.data_; }
    SInt operator^ (SInt other) const { return this->data_ ^ other.data_; }
    SInt operator>>(SInt other) const { return this->data_ >>other.data_; }
    bool operator< (SInt other) const { return this->data_ < other.data_; }
    bool operator<=(SInt other) const { return this->data_ <=other.data_; }
    bool operator> (SInt other) const { return this->data_ > other.data_; }
    bool operator>=(SInt other) const { return this->data_ >=other.data_; }
    bool operator==(SInt other) const { return this->data_ ==other.data_; }
    bool operator!=(SInt other) const { return this->data_ !=other.data_; }
    bool is_neg() const { return data_ < ST(0); }
    operator ST() const { return data_; }
    ST data() const { return data_; }

private:
    SInt minus() const { return SInt(~UT(data_)+UT(1u)); }
    SInt abs() const { return is_neg() ? minus() : *this; }

    ST data_;
};

template<class UT, bool wrap>
class UInt {
public:
    static_assert(std::is_unsigned<UT>::value, "UT must be an unsigned type");

    UInt() = default;
    UInt(UT data)
        : data_(data)
    {}

    UInt operator-() const { return UInt(0u) - *this; }

    UInt operator+(UInt other) const {
        UInt res(UT(this->data_) + UT(other.data_));
        if (!wrap && res.data_ < this->data_)
            throw BottomException();
        return res;
    }

    UInt operator-(UInt other) const {
        UInt res(UT(this->data_) - UT(other.data_));
        if (!wrap && res.data_ > this->data_)
            throw BottomException();
        return res;
    }

    UInt operator*(UInt other) const {
        if (!wrap && other.data_ && this->data_ > std::numeric_limits<UT>::max() / other.data_)
            throw BottomException();
        return UT(this->data_) * UT(other.data_);
    }

    void div_check(UInt other) const { if (other.data_ == UT(0u)) throw BottomException(); }
    UInt operator/(UInt other) const { div_check(other); return UInt(this->data_ / other.data_); }
    UInt operator%(UInt other) const { div_check(other); return UInt(this->data_ % other.data_); }

    UInt operator<<(UInt other) const {
        if (!wrap && other.data_ >= std::numeric_limits<UT>::digits)
            throw BottomException();
        return this->data_ << other.data_;
    }

    UInt operator& (UInt other) const { return this->data_ & other.data_; }
    UInt operator| (UInt other) const { return this->data_ | other.data_; }
    UInt operator^ (UInt other) const { return this->data_ ^ other.data_; }
    UInt operator>>(UInt other) const { return this->data_ >>other.data_; }
    bool operator< (UInt other) const { return this->data_ < other.data_; }
    bool operator<=(UInt other) const { return this->data_ <=other.data_; }
    bool operator> (UInt other) const { return this->data_ > other.data_; }
    bool operator>=(UInt other) const { return this->data_ >=other.data_; }
    bool operator==(UInt other) const { return this->data_ ==other.data_; }
    bool operator!=(UInt other) const { return this->data_ !=other.data_; }
    bool is_neg() const { return data_ < UT(0u); }
    operator UT() const { return data_; }
    UT data() const { return data_; }

private:
    UInt minus() const { return UInt(~UT(data_)+UT(1u)); }
    UInt abs() const { return is_neg() ? minus() : *this; }

    UT data_;
};

inline half        rem(half a, half b)               { return      fmod(a, b); }
inline float       rem(float a, float b)             { return std::fmod(a, b); }
inline double      rem(double a, double b)           { return std::fmod(a, b); }
inline long double rem(long double a, long double b) { return std::fmod(a, b); }

template<class FT, bool precise>
class Float {
public:
    Float() = default;
    Float(FT data)
        : data_(data)
    {}

    Float operator- () const { return -data_; }
    Float operator+ (Float other) const { return Float(this->data_ + other.data_); }
    Float operator- (Float other) const { return Float(this->data_ - other.data_); }
    Float operator* (Float other) const { return Float(this->data_ * other.data_); }
    Float operator/ (Float other) const { return Float(this->data_ / other.data_); }
    Float operator% (Float other) const { return Float(rem(this->data_, other.data_)); }
    bool  operator< (Float other) const { return this->data_ < other.data_; }
    bool  operator<=(Float other) const { return this->data_ <=other.data_; }
    bool  operator> (Float other) const { return this->data_ > other.data_; }
    bool  operator>=(Float other) const { return this->data_ >=other.data_; }
    bool  operator==(Float other) const { return this->data_ ==other.data_; }
    bool  operator!=(Float other) const { return this->data_ !=other.data_; }
    operator FT() const { return data_; }
    FT data() const { return data_; }

private:
    FT data_;
};

template<class FT, bool precise>
std::ostream& operator<<(std::ostream& os, const Float<FT, precise>& ft) { return os << ft.data(); }

typedef  int8_t  s8; typedef  uint8_t  u8; typedef SInt< s8, true>  ps8; typedef UInt< u8, true>  pu8; typedef SInt< s8, false>  qs8; typedef UInt< u8, false>  qu8;
typedef int16_t s16; typedef uint16_t u16; typedef SInt<s16, true> ps16; typedef UInt<u16, true> pu16; typedef SInt<s16, false> qs16; typedef UInt<u16, false> qu16;

typedef int32_t s32; typedef uint32_t u32; typedef SInt<s32, true> ps32; typedef UInt<u32, true> pu32; typedef SInt<s32, false> qs32; typedef UInt<u32, false> qu32;
typedef int64_t s64; typedef uint64_t u64; typedef SInt<s64, true> ps64; typedef UInt<u64, true> pu64; typedef SInt<s64, false> qs64; typedef UInt<u64, false> qu64;

typedef half   f16; typedef Float<f16, true> pf16; typedef Float<f16, false> qf16;
typedef float  f32; typedef Float<f32, true> pf32; typedef Float<f32, false> qf32;
typedef double f64; typedef Float<f64, true> pf64; typedef Float<f64, false> qf64;

}

#endif
