#include "impala/type.h"

#include "anydsl2/printer.h"

using anydsl2::Pi;

namespace impala {

World::World() 
    : noret_(keep(new NoRet(*this)))
    , type_void_(keep(new Void(*this)))
    , type_error_(keep(new TypeError(*this)))
{}

const anydsl2::Type* return_type(const anydsl2::Pi* pi) {
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

void NoRet::vdump(anydsl2::Printer& p) const { p << "noret"; }
void TypeError::vdump(anydsl2::Printer& p) const { p << "<type error>"; }
void Void::vdump(anydsl2::Printer& p) const { p << "void"; }

} // namespace impala
