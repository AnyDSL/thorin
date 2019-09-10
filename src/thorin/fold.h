#ifndef THORIN_FOLD_H
#define THORIN_FOLD_H

#include <optional>

#include "thorin/tables.h"
#include "thorin/util/cast.h"
#include "thorin/util/types.h"

namespace thorin::fold {

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

template<WOp> struct FoldWOp {};

template<> struct FoldWOp<WOp::add> {
    template<nat_t w> struct Fold {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
            auto x = get<w2u<w>>(a);
            auto y = get<w2u<w>>(b);
            decltype(x) res = x + y;
            if (nuw && res < x) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct FoldWOp<WOp::sub> {
    template<nat_t w> struct Fold {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool /*nuw*/) {
            typedef w2u<w> UT;
            auto x = get<UT>(a);
            auto y = get<UT>(b);
            decltype(x) res = x - y;
            //if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct FoldWOp<WOp::mul> {
    template<nat_t w> struct Fold {
        static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
            typedef w2u<w> UT;
            auto x = get<UT>(a);
            auto y = get<UT>(b);
            decltype(x) res = x * y;
            if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
            // TODO nsw
            return res;
        }
    };
};

template<> struct FoldWOp<WOp::shl> {
    template<nat_t w> struct Fold {
        static Res run(u64 aa, u64 bb, bool nsw, bool nuw) {
            typedef w2u<w> T;
            auto a = get<T>(aa);
            auto b = get<T>(bb);
            if (b > w) return {};
            decltype(a) res = a << b;
            if (nuw && res < a) return {};
            if (nsw && get_sign(a) != get_sign(res)) return {};
            return res;
        }
    };
};

template<ZOp> struct FoldZOp {};
template<> struct FoldZOp<ZOp::sdiv> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct FoldZOp<ZOp::udiv> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct FoldZOp<ZOp::smod> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };
template<> struct FoldZOp<ZOp::umod> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };

template<IOp> struct FoldIOp {};
template<> struct FoldIOp<IOp::ashr> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2s<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct FoldIOp<IOp::lshr> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct FoldIOp<IOp::iand> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) & get<T>(b)); } }; };
template<> struct FoldIOp<IOp::ior > { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) | get<T>(b)); } }; };
template<> struct FoldIOp<IOp::ixor> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) ^ get<T>(b)); } }; };

template<ROp> struct FoldROp {};
template<> struct FoldROp<ROp::add> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2r<w>; return w2r<w>(get<w2r<w>>(a) + get<T>(b)); } }; };
template<> struct FoldROp<ROp::sub> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) - get<T>(b)); } }; };
template<> struct FoldROp<ROp::mul> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) * get<T>(b)); } }; };
template<> struct FoldROp<ROp::div> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) / get<T>(b)); } }; };
template<> struct FoldROp<ROp::mod> { template<nat_t w> struct Fold { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(rem(get<T>(a), get<T>(b))); } }; };

template<ICmp cmp> struct FoldICmp {
    template<nat_t w> struct Fold {
        inline static Res run(u64 a, u64 b) {
            typedef w2u<w> T;
            auto x = get<T>(a);
            auto y = get<T>(b);
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

template<RCmp cmp> struct FoldRCmp {
    template<nat_t w> struct Fold {
        inline static Res run(u64 a, u64 b) {
            typedef w2r<w> T;
            auto x = get<T>(a);
            auto y = get<T>(b);
            bool result = false;
            result |= ((cmp & RCmp::u) != RCmp::f) && std::isunordered(x, y);
            result |= ((cmp & RCmp::g) != RCmp::f) && x > y;
            result |= ((cmp & RCmp::l) != RCmp::f) && x < y;
            result |= ((cmp & RCmp::e) != RCmp::f) && x == y;
            return result;
        }
    };
};

template<I2I> struct FoldI2I {};
template<> struct FoldI2I<I2I::s2s> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2s<dw>(get<w2s<sw>>(src)); } }; };
template<> struct FoldI2I<I2I::u2u> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2u<dw>(get<w2u<sw>>(src)); } }; };

template<I2R> struct FoldI2R {};
template<> struct FoldI2R<I2R::s2r> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2r<dw>(get<w2s<sw>>(src)); } }; };
template<> struct FoldI2R<I2R::u2r> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2r<dw>(get<w2u<sw>>(src)); } }; };

template<R2I> struct FoldR2I {};
template<> struct FoldR2I<R2I::r2s> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2s<dw>(get<w2r<sw>>(src)); } }; };
template<> struct FoldR2I<R2I::r2u> { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2u<dw>(get<w2r<sw>>(src)); } }; };

struct FoldR2R { template<nat_t sw, nat_t dw> struct Fold { static Res run(u64 src) { return w2r<dw>(get<w2r<sw>>(src)); } }; };

}

#endif
