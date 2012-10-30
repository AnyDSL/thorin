#ifndef ANYDSL2_TYPE_H
#define ANYDSL2_TYPE_H

#include <exception>
#include <iterator>

#include "anydsl2/node.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {

class Def;
class Generic;
class Lambda;
class Pi;
class PrimLit;
class Printer;
class Ptr;
class Type;
class World;

typedef ArrayRef<const Type*> Elems;

//------------------------------------------------------------------------------

class type_error : public std::exception {
public:

    type_error(const Type* type1, const Type* type2)
        : type1_(type1)
        , type2_(type2)
    {}

    const Type* type1() const { return type1_; }
    const Type* type2() const { return type2_; }
    virtual const char* what() const throw();

private:

    const Type* type1_;
    const Type* type2_;
};

class inference_exception : public std::exception {
public:

    inference_exception(const Generic* generic, const Type* expected, const Type* found)
        : generic_(generic)
        , expected_(expected)
        , found_(found)
    {}

    const Generic* generic() const { return generic_; }
    const Type* expected() const { return expected_; }
    const Type* found() const { return found_; }
    virtual const char* what() const throw();

private:

    const Generic* generic_;
    const Type* expected_;
    const Type* found_;
};

class GenericMap : protected std::vector<const Type*> {
public:

    GenericMap() {}

    const Type*& operator [] (const Generic* generic);
    bool is_empty() const;
    const char* to_string() const;

private:
    const Type*& get(size_t i) { return std::vector<const Type*>::operator[](i); }
    const Type* const & get(size_t i) const { return std::vector<const Type*>::operator[](i); }
};

inline std::ostream& operator << (std::ostream& o, const GenericMap& map) { 
    o << map.to_string(); return o; 
}

//------------------------------------------------------------------------------

class Type : public Node {
protected:

    Type(World& world, int kind, size_t num)
        : Node(kind, num)
        , world_(world)
    {}

public:

    void dump() const;
    void dump(bool fancy) const;
    World& world() const { return world_; }
    Elems elems() const { return ops_ref<const Type*>(); }
    const Type* elem(size_t i) const { return elems()[i]; }
    const Type* elem_via_lit(const Def* def) const;
    const Ptr* to_ptr() const;
    virtual void vdump(Printer &printer) const = 0;
    void infer(GenericMap& map, const Type* type) const;
    GenericMap infer(const Type* type) const {
        GenericMap map;
        infer(map, type);
        return map;
    }

private:

    World& world_;

    friend class Def;
};

std::ostream& operator << (std::ostream& o, const anydsl2::Type* type);

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

    const Type* ref() const { return elem(0); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_elems);
    CompoundType(World& world, int kind, ArrayRef<const Type*> elems);

    void dump_inner(Printer& printer) const;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t size)
        : CompoundType(world, Node_Sigma, size)
        , named_(true)
    {}
    Sigma(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Sigma, elems)
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

    Pi(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Pi, elems)
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

    Generic(World& world, size_t index)
        : Type(world, Node_Generic, 0)
        , index_(index)
    {}

public:

    virtual void vdump(Printer& printer) const;
    size_t index() const { return index_; }

private:

    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

    size_t index_;

    friend class World;
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

class GenericBuilder {
public:

    GenericBuilder(World& world)
        : world_(world)
        , index_(0)
    {}

    size_t new_def();
    const Generic* use(size_t handle);
    void pop();

private:

    World& world_;
    size_t index_;
    typedef std::vector<const Generic*> Index2Generic;
    Index2Generic index2generic_;
};

//------------------------------------------------------------------------------

bool check(const Type* t1, const Type* t2);

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
