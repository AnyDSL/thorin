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
    Type_Void,
    Type_Error,
};

class Void : public anydsl2::Type {
private:
    Void(anydsl2::World& world) 
        : anydsl2::Type(world, Type_Void, 0, false)
    {}

    friend class World;
};

class NoRet : public anydsl2::Type {
private:
    NoRet(anydsl2::World& world) 
        : anydsl2::Type(world, Type_NoRet, 0, false)
    {}

    friend class World;
};

class TypeError : public anydsl2::Type {
private:
    TypeError(anydsl2::World& world) 
        : anydsl2::Type(world, Type_Error, 0, false)
    {}

    friend class World;
};

class World : public anydsl2::World {
public:
    World();

    const NoRet* noret() { return noret_; }
    const Void* type_void() { return type_void_; }
    const TypeError* type_error() { return type_error_; }

private:
    const NoRet* noret_;
    const Void* type_void_;
    const TypeError* type_error_;
};

const anydsl2::Type* return_type(const anydsl2::Pi*);
const anydsl2::Type* convert_type(const anydsl2::Type*);
template<class T>
const T* convert(const T* type) { return convert_type(type)->template as<T>(); }

} // namespace impala

#endif
