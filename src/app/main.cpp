//#include <iostream>

#include "anydsl/air/binop.h"
#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/terminator.h"
#include "anydsl/air/lambda.h"
#include "anydsl/air/world.h"
#include "anydsl/util/box.h"
#include "anydsl/util/cast.h"
#include "anydsl/util/location.h"
#include "anydsl/util/ops.h"
#include "anydsl/util/types.h"
#include "anydsl/util/foreach.h"

using namespace anydsl;

int main() {
    std::cout << Location(Position("aaa", 23, 42), Position("bbb", 101, 666)) << std::endl;
    std::cout << Location(Position("aaa", 23, 42), Position("aaa", 101, 666)) << std::endl;
    std::cout << Location(Position("aaa", 23, 42), Position("aaa", 23, 666)) << std::endl;
    std::cout << Location(Position("aaa", 23, 42), Position("aaa", 23, 42)) << std::endl;

    std::cout << std::endl;

    std::cout << Location("aaa:23 col 42 - bbb:101 col 666") << std::endl;
    std::cout << Location("aaa:23 col 42 - 101 col 666") << std::endl;
    std::cout << Location("aaa:23 col 42 - 666") << std::endl;
    std::cout << Location("aaa:23 col 42") << std::endl;

    std::cout << std::endl;

    std::cout << "+" << std::endl;
    std::cout << u1(false) + u1(false) << std::endl;
    std::cout << u1(true) + u1(false) << std::endl;
    std::cout << u1(false) + u1(true) << std::endl;
    std::cout << u1(true) + u1(true) << std::endl;

    std::cout << "*" << std::endl;
    std::cout << u1(false) * u1(false) << std::endl;
    std::cout << u1(true) * u1(false) << std::endl;
    std::cout << u1(false) * u1(true) << std::endl;
    std::cout << u1(true) * u1(true) << std::endl;

    std::cout << "^" << std::endl;
    std::cout << (u1(false) ^ u1(false)) << std::endl;
    std::cout << (u1(true) ^ u1(false)) << std::endl;
    std::cout << (u1(false) ^ u1(true)) << std::endl;
    std::cout << (u1(true) ^ u1(true)) << std::endl;

    {
        std::cout << "+=" << std::endl;
        u1 u(false);
        std::cout << u << std::endl;
        u += 1;
        std::cout << u << std::endl;
        u += 1;
        std::cout << u << std::endl;
    }

    {
        std::cout << "pre ++" << std::endl;
        u1 u(false);
        std::cout << u << std::endl;
        ++u;
        std::cout << u << std::endl;
        ++u;
        std::cout << u << std::endl;
    }

    std::cout << std::endl;
    std::cout << Num_Nodes << std::endl;
    std::cout << Num_ArithOps << std::endl;
    std::cout << Num_RelOps << std::endl;
    std::cout << Num_ConvOps << std::endl;
    std::cout << Num_Indexes << std::endl;

    World w;
    std::cout << w.type_u8()->debug << std::endl;

    std::cout << std::endl;

    std::cout << w.type(anydsl::PrimType_u1)->debug << std::endl;
    std::cout << w.type(anydsl::PrimType_u64)->debug << std::endl;
    std::cout << w.type(anydsl::PrimType_f32)->debug << std::endl;
    std::cout << w.type(anydsl::PrimType_f64)->debug << std::endl;
    std::cout << w.type_f64()->debug << std::endl;

    Sigma* s = w.sigma();
    const Type* members[4] = {w.type_u8(), w.type_f32(), w.type_u1(), w.type_u8()};
    s->set(members, members + 4);

    Lambda* l = w.createLambda(0);
    l->debug = "hallo";
    Goto* g = w.createGoto(l, l);
    Args& args = g->jump.args;
    args.append(w.literal(7u))->debug = "7";
    args.append(w.literal(32ul))->debug = "32";
    Args::iterator i = args.append(w.literal(666ul)); i->debug = "666";
    args.append(w.literal(64ul))->debug = "64";
    args.prepend(w.literal(1u))->debug = "1";
    l->appendParam(w.type_u8());
    l->appendParam(w.type_u8());
    l->appendParam(w.type_u32());

    w.createLambda(w.pi((boost::array<const Type*, 3>){{ w.type_u8(), w.type_u8(), w.type_u32()}}));

    // prepend
    // remove evil and substitute by heavenly
    std::cout << "evil: " << i->debug << std::endl;
    i = args.erase(i);
    args.insert(i, w.literal(777ul))->debug = "777";

    std::cout << "--- testing args ---" << std::endl;
    FOREACH(const& use, args)
        std::cout << "--> " << use.debug << std::endl;

    std::cout << "--- reverse args ---" << std::endl;
    for (Args::const_reverse_iterator i = args.rbegin(), e = args.rend(); i != e; ++i)
        std::cout << "--> " << i->debug << std::endl;

    return 0;
}
