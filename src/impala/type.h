#ifndef IMPALA_TYPE_H
#define IMPALA_TYPE_H

#include "anydsl2/type.h"
#include "anydsl2/world.h"

namespace anydsl2 {
    class CodeGen;
    class Printer;
}

namespace impala {

class World;

enum {
    Type_NoRet = anydsl2::End_AllNodes,
    Type_Error,
};

class NoRet : public anydsl2::Type {
private:

    NoRet(anydsl2::World& world) 
        : anydsl2::Type(world, Type_NoRet, 0)
    {}

    virtual void vdump(anydsl2::Printer& printer) const { ANYDSL2_UNREACHABLE; }

    friend class World;
};

class TypeError : public anydsl2::Type {
private:

    TypeError(anydsl2::World& world) 
        : anydsl2::Type(world, Type_Error, 0)
    {}

    virtual void vdump(anydsl2::Printer& printer) const { ANYDSL2_UNREACHABLE; }

    friend class World;
};

class World : public anydsl2::World {
public:

    World();

    const NoRet* noret() { return noret_; }
    const TypeError* type_error() { return error_; }

private:

    const NoRet* noret_;
    const TypeError* error_;
};

const anydsl2::Type* return_type(const anydsl2::Pi* pi);

} // namespace impala

#endif
