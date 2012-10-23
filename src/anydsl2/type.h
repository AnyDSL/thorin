#ifndef ANYDSL2_TYPE_H
#define ANYDSL2_TYPE_H

#include <iterator>

#include "anydsl2/node.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {

class Generic;
class Lambda;
class Pi;
class PrimLit;
class Printer;
class Ptr;
class Type;
class World;

//------------------------------------------------------------------------------

class Type : public Node {
protected:

    Type(World& world, int kind, size_t num)
        : Node(kind, num)
        , world_(world)
    {}

public:

    typedef ArrayRef<const Type*> Args;

    void dump() const;
    void dump(bool fancy) const;
    World& world() const { return world_; }
    Args args() const { return ops_ref<const Type*>(); }
    const Type* arg(size_t i) const { return args()[i]; }
    const Ptr* to_ptr() const;
    virtual void vdump(Printer &printer) const = 0;

private:

    World& world_;

    friend class Def;
};

//------------------------------------------------------------------------------

/// The type of the memory monad.
class Mem : public Type {
private:

    Mem(World& world)
        : Type(world, Node_Mem, 0)
    {}
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// The type of a stack frame.
class Frame : public Type {
private:

    Frame(World& world)
        : Type(world, Node_Frame, 0)
    {}
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind);

public:

    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

    bool is_int()   const { return anydsl2::is_int(primtype_kind()); }
    bool is_float() const { return anydsl2::is_float(primtype_kind()); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Ptr : public Type {
private:

    Ptr(const Type* ref)
        : Type(ref->world(), Node_Ptr, 1)
    {
        set(0, ref);
    }

    virtual void vdump(Printer& printer) const;

public:

    const Type* ref() const { return arg(0); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_generics, size_t num_elems);
    CompoundType(World& world, int kind, ArrayRef<const Generic*> generics, 
                                         ArrayRef<const Type*> elems);
    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

    void dump_inner(Printer& printer) const;

public:

    typedef ArrayRef<const Type*> Elems;
    typedef ArrayRef<const Generic*> Generics;

    Elems elems() const { return ops_ref<const Type*>().slice_back(num_generics_); }
    const Type* elem(size_t i) const { return elems()[i]; }

    Generics generics() const { return ops_ref<const Generic*>().slice_front(num_generics_); }
    const Generic* generic(size_t i) const { return generics()[i]; }

    size_t num_generics() const { return num_generics_; }
    size_t num_elems() const { return size() - num_generics(); }

private:

    size_t num_generics_;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t num_elems, size_t num_generics)
        : CompoundType(world, Node_Sigma, num_elems, num_generics)
        , named_(true)
    {}
    Sigma(World& world, ArrayRef<const Generic*> generics, 
                        ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Sigma, generics, elems)
        , named_(false)
    {}

public:

    bool named() const { return named_; }

private:

    virtual void vdump(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world, ArrayRef<const Generic*> generics, 
                     ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Pi, generics, elems)
    {}

public:

    bool is_fo() const;
    bool is_ho() const;

private:

    template<bool fo> bool classify_order() const;
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Generic : public Type {
private:

    Generic(Lambda* lambda, size_t index);

public:

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }

    virtual void vdump(Printer& printer) const;

private:

    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

    Lambda* lambda_;
    size_t index_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

inline bool is_u1(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return p->primtype_kind() == PrimType_u1;
    return false;
}

inline bool is_int(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return is_int(p->primtype_kind());
    return false;
}

inline bool is_float(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return is_float(p->primtype_kind());
    return false;
}

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
