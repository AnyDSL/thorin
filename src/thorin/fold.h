#ifndef THORIN_FOLD_H
#define THORIN_FOLD_H

#include <optional>

#include "thorin/tables.h"
#include "thorin/util/cast.h"
#include "thorin/util/types.h"

namespace thorin {

// This code assumes two-complement arithmetic for unsigned operations.
// This is *implementation-defined* but *NOT* *undefined behavior*.

class Res {
public:
    Res()
        : data_{}
    {}
    template<class T>
    Res(T val)
        : data_(bitcast<u64>(val))
    {}

    constexpr const u64& operator*() const& { return *data_; }
    constexpr u64& operator*() & { return *data_; }
    explicit operator bool() const { return data_.has_value(); }

private:
    std::optional<u64> data_;
};

template<class T> inline T get(u64 u) { return bitcast<T>(u); }

template<class T, T> struct Fold {};

template<> struct Fold<WOp, WOp::add> {
    template<nat_t w> struct F {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
            auto x = get<w2u<w>>(a), y = get<w2u<w>>(b);
            decltype(x) res = x + y;
            if (nuw && res < x) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct Fold<WOp, WOp::sub> {
    template<nat_t w> struct F {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool /*nuw*/) {
            using UT = w2u<w>;
            auto x = get<UT>(a), y = get<UT>(b);
            decltype(x) res = x - y;
            //if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct Fold<WOp, WOp::mul> {
    template<nat_t w> struct F {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
            using UT = w2u<w>;
            auto x = get<UT>(a), y = get<UT>(b);
            decltype(x) res = x * y;
            if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct Fold<WOp, WOp::shl> {
    template<nat_t w> struct F {
        static Res run(u64 a, u64 b, bool nsw, bool nuw) {
            using T = w2u<w>;
            auto x = get<T>(a), y = get<T>(b);
            if (y > w) return {};
            decltype(x) res = x << y;
            if (nuw && res < x) return {};
            if (nsw && get_sign(x) != get_sign(res)) return {};
            return res;
        }
    };
};

template<> struct Fold<ZOp, ZOp::sdiv> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct Fold<ZOp, ZOp::udiv> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct Fold<ZOp, ZOp::smod> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };
template<> struct Fold<ZOp, ZOp::umod> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };

template<> struct Fold<IOp, IOp::ashr> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2s<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct Fold<IOp, IOp::lshr> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct Fold<IOp, IOp::iand> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) & get<T>(b)); } }; };
template<> struct Fold<IOp, IOp::ior > { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) | get<T>(b)); } }; };
template<> struct Fold<IOp, IOp::ixor> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) ^ get<T>(b)); } }; };

template<> struct Fold<ROp, ROp::add> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2r<w>; return w2r<w>(get<w2r<w>>(a) + get<T>(b)); } }; };
template<> struct Fold<ROp, ROp::sub> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) - get<T>(b)); } }; };
template<> struct Fold<ROp, ROp::mul> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) * get<T>(b)); } }; };
template<> struct Fold<ROp, ROp::div> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) / get<T>(b)); } }; };
template<> struct Fold<ROp, ROp::mod> { template<nat_t w> struct F { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(rem(get<T>(a), get<T>(b))); } }; };

template<ICmp cmp> struct Fold<ICmp, cmp> {
    template<nat_t w> struct F {
        inline static Res run(u64 a, u64 b) {
            using T = w2u<w>;
            auto x = get<T>(a), y = get<T>(b);
            bool result = false;
            auto pm = !(x >> T(w-1)) &&  (y >> T(w-1));
            auto mp =  (x >> T(w-1)) && !(y >> T(w-1));
            result |= ((cmp & ICmp::_x) != ICmp::_f) && pm;
            result |= ((cmp & ICmp::_y) != ICmp::_f) && mp;
            result |= ((cmp & ICmp::_g) != ICmp::_f) && x > y && !mp;
            result |= ((cmp & ICmp::_l) != ICmp::_f) && x < y && !pm;
            result |= ((cmp & ICmp:: e) != ICmp::_f) && x == y;
            return result;
        }
    };
};

template<RCmp cmp> struct Fold<RCmp, cmp> {
    template<nat_t w> struct F {
        inline static Res run(u64 a, u64 b) {
            using T = w2r<w>;
            auto x = get<T>(a), y = get<T>(b);
            bool result = false;
            result |= ((cmp & RCmp::u) != RCmp::f) && std::isunordered(x, y);
            result |= ((cmp & RCmp::g) != RCmp::f) && x > y;
            result |= ((cmp & RCmp::l) != RCmp::f) && x < y;
            result |= ((cmp & RCmp::e) != RCmp::f) && x == y;
            return result;
        }
    };
};

template<> struct Fold<Conv, Conv::s2s> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2s<dw>(get<w2s<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::u2u> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2u<dw>(get<w2u<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::s2r> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2r<dw>(get<w2s<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::u2r> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2r<dw>(get<w2u<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::r2s> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2s<dw>(get<w2r<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::r2u> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2u<dw>(get<w2r<sw>>(src)); } }; };
template<> struct Fold<Conv, Conv::r2r> { template<nat_t sw, nat_t dw> struct F { static Res run(u64 src) { return w2r<dw>(get<w2r<sw>>(src)); } }; };

}

#endif
