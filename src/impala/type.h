#ifndef IMPALA_TYPE_H
#define IMPALA_TYPE_H

#include "impala/token.h"

#include <boost/unordered_set.hpp>

namespace anydsl {
    class Type;
    class World;
}

namespace impala {

class Printer;

class Type {
public:

    virtual ~Type() {}

    virtual void dump(Printer& p) const = 0;
    virtual const anydsl::Type* emit(anydsl::World& world) const = 0;

    virtual bool isBool() const { return false; }
    virtual bool isError() const { return false; }
    virtual bool isVoid() const { return false; }

    ANYDSL_MIXIN_AS_ISA

private:

    virtual bool equal(const Type* t) const = 0;
    virtual size_t hash() const = 0;

    friend class TypeHash;
    friend class TypeEqual;
};

class PrimType : public Type {
public:

    enum Kind {
#define IMPALA_TYPE(itype, atype) TYPE_##itype = Token:: TYPE_##itype,
#include "impala/tokenlist.h"
    };

private:

    PrimType();
    PrimType(const PrimType&);
    PrimType(Kind kind)
        : kind_(kind)
    {}

public:

    Kind kind() const { return kind_; }

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* emit(anydsl::World& world) const;
    virtual bool isBool() const { return kind_ == TYPE_bool; }

private:

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;

    Kind kind_;

    friend class TypeTable;
};

class Void : public Type {
public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* emit(anydsl::World& world) const;

private:

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;
};

class TypeError : public Type {
public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* emit(anydsl::World& world) const;
    virtual bool isError() const { return true; }

private:

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;
};

//------------------------------------------------------------------------------

struct TypeHash : std::unary_function<const Type*, size_t> {
    size_t operator () (const Type* t) const { return t->hash(); }
};

struct TypeEqual : std::binary_function<const Type*, const Type*, bool> {
    bool operator () (const Type* t1, const Type* t2) const { return t1->equal(t2); }
};

class TypeTable {
public:

    TypeTable();
    ~TypeTable();

    const PrimType* type(PrimType::Kind kind);
#define IMPALA_TYPE(itype, atype) \
    const PrimType* type_##itype() { return itype##_; }
#include "impala/tokenlist.h"

    const TypeError* type_error() const { return type_error_; }
    const Void* type_void() const { return type_void_; }

    typedef boost::unordered_set<const Type*, TypeHash, TypeEqual> TypeSet;
    TypeSet types_;

private:

    const TypeError* type_error_;
    const Void* type_void_;
#define IMPALA_TYPE(itype, atype) const PrimType* itype##_;
#include "impala/tokenlist.h"
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_TYPE_H
