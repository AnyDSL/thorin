#ifndef THORIN_FOLD_H
#define THORIN_FOLD_H

#include <optional>

#include "thorin/tables.h"
#include "thorin/util/cast.h"
#include "thorin/util/types.h"

namespace thorin::fold {

// This code assumes two-complement arithmetic for unsigned operations.
// This is *implementation-defined* but *NOT* *undefined behavior*.

using Res = std::optional<u64>;

template<class T>
inline T get(u64 u) { return bitcast<T>(u); }

template<WOp> struct FoldWOp {};

template<> struct FoldWOp<WOp::add> {
    template<int w> struct Fold {
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
    template<int w> struct Fold {
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
    template<int w> struct Fold {
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
    template<int w> struct Fold {
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
template<> struct FoldZOp<ZOp::sdiv> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2s<w> T; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct FoldZOp<ZOp::udiv> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } }; };
template<> struct FoldZOp<ZOp::smod> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2s<w> T; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };
template<> struct FoldZOp<ZOp::umod> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } }; };

template<IOp> struct FoldIOp {};
template<> struct FoldIOp<IOp::ashr> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2s<w> T; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct FoldIOp<IOp::lshr> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } }; };
template<> struct FoldIOp<IOp::iand> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; return T(get<T>(a) & get<T>(b)); } }; };
template<> struct FoldIOp<IOp::ior > { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; return T(get<T>(a) | get<T>(b)); } }; };
template<> struct FoldIOp<IOp::ixor> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2u<w> T; return T(get<T>(a) ^ get<T>(b)); } }; };

template<ROp> struct FoldROp {};
template<> struct FoldROp<ROp::add> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2r<w> T; return w2r<w>(get<w2r<w>>(a) + get<T>(b)); } }; };
template<> struct FoldROp<ROp::sub> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2r<w> T; return T(get<T>(a) - get<T>(b)); } }; };
template<> struct FoldROp<ROp::mul> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2r<w> T; return T(get<T>(a) * get<T>(b)); } }; };
template<> struct FoldROp<ROp::div> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2r<w> T; return T(get<T>(a) / get<T>(b)); } }; };
template<> struct FoldROp<ROp::mod> { template<int w> struct Fold { static Res run(u64 a, u64 b) { typedef w2r<w> T; return T(rem(get<T>(a), get<T>(b))); } }; };

template<ICmp cmp> struct FoldICmp {
    template<int w> struct Fold {
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
    template<int w> struct Fold {
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

}

#endif
