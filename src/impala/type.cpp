#include "impala/type.h"

#include "anydsl2/printer.h"

using namespace anydsl2;

namespace impala {

World::World() 
    : noret_(keep(new NoRet(*this)))
    , type_void_(keep(new Void(*this)))
    , type_error_(keep(new TypeError(*this)))
{}

const Type* return_type(const Pi* pi) {
    if (!pi->empty()) {
        if (const Pi* ret = pi->elems().back()->isa<Pi>()) {
            if (ret->size() == 1)
                return ret->elem(0);
            if (ret->size() == 0)
                return ((impala::World&) pi->world()).type_void();
        }
    }

    return ((impala::World&) pi->world()).noret();
}

Printer& NoRet::print(Printer& p) const { p << "noret"; return p; }
Printer& TypeError::print(Printer& p) const { p << "<type error>"; return p; }
Printer& Void::print(Printer& p) const { p << "void"; return p; }

const Type* convert_type(const Type* type) {
    if (const Pi* pi = type->isa<Pi>()) {
        Array<const Type*> elems(pi->size() + 1);
        elems[0] = type->world().mem();

        for (size_t i = 1, e = elems.size(); i != e; ++i)
            elems[i] = convert(pi->elem(i-1));

        return type->world().pi(elems);
    }

    return type;
}

} // namespace impala
