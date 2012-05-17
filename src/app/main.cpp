//#include <iostream>

#include "anydsl/air/binop.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/type.h"
#include "anydsl/support/world.h"
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

    //ArithOp* p = new ArithOp(ArithOp_add, 0, 0, "todo");
    //switch (p->arithOpKind()) {
        //case ArithOp_ashr
    //}

    //std::cout << p-> << std::endl;
    //

    std::cout << std::endl;
    std::cout << Num_Nodes << std::endl;
    std::cout << Num_ArithOps << std::endl;
    std::cout << Num_RelOps << std::endl;
    std::cout << Num_ConvOps << std::endl;
    std::cout << Num_Indexes << std::endl;

    World w;
    std::cout << w.type_u8()->debug() << std::endl;

    std::cout << std::endl;

    std::cout << w.type(anydsl::PrimType_u1)->debug() << std::endl;
    std::cout << w.type(anydsl::PrimType_u64)->debug() << std::endl;
    std::cout << w.type(anydsl::PrimType_f32)->debug() << std::endl;
    std::cout << w.type(anydsl::PrimType_f64)->debug() << std::endl;
    std::cout << w.type_f64()->debug() << std::endl;

    Sigma* s = w.getNamedSigma();
    const Type* members[4] = {w.type_u8(), w.type_f32(), w.type_u1(), w.type_u8()};
    s->set(members, members + 4);

    Args args(0);
    args.append(w.constant(7u), "7");
    args.append(w.constant(32ul), "32");

    std::cout << "testing args" << std::endl;
    FOREACH(const& use, args)
        std::cout << "--> " << use.debug() << std::endl;

    //return 0;
}
