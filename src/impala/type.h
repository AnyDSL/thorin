#ifndef IMPALA_TYPE_H
#define IMPALA_TYPE_H

#include <boost/unordered_set.hpp>

#include "anydsl/symbol.h"
#include "anydsl/util/array.h"

#include "impala/token.h"

namespace anydsl {
    class Type;
    class World;
}

namespace impala {

class CodeGen;
class Fct;
class Printer;

class Type : public anydsl::MagicCast {
public:

    virtual ~Type() {}

    virtual void dump(Printer& p) const = 0;
    virtual const anydsl::Type* convert(CodeGen&) const = 0;

    virtual bool is_bool() const { return false; }
    virtual bool is_error() const { return false; }
    virtual bool is_int() const { return false; }
    virtual bool is_noret() const { return false; }
    virtual bool is_void() const { return false; }
    virtual bool is_generic() const { return false; }

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

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;

public:

    Kind kind() const { return kind_; }

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;
    virtual bool is_bool() const { return kind_ == TYPE_bool; }
    virtual bool is_int() const;

private:

    Kind kind_;

    friend class TypeTable;
};

class Void : public Type {
private:

    Void() {}

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;
    virtual bool is_void() const { return true; }

    friend class TypeTable;
};

class NoRet : public Type {
private:

    NoRet() {}

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;
    virtual bool is_noret() const { return true; }

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;
    virtual bool is_void() const { return true; }

    friend class TypeTable;
};

class TypeError : public Type {
private:

    TypeError() {}

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;
    virtual bool is_error() const { return true; }

    friend class TypeTable;
};

class Pi : public Type {
private:

    Pi(anydsl::ArrayRef<const Type*> elems, const Type* ret);

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;

    typedef anydsl::ArrayRef<const Type*> Elems;
    Elems elems() const { return Elems(elems_); }
    const Type* ret() const { return ret_; }

private:

    virtual bool equal(const Type* other) const;
    virtual size_t hash() const;

    anydsl::Array<const Type*> elems_;
    const Type* ret_;

    friend class TypeTable;
};

class Sigma : public Type {
private:

    Sigma(anydsl::ArrayRef<const Type*> elems);

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;

    typedef anydsl::ArrayRef<const Type*> Elems;
    Elems elems() const { return Elems(elems_); }
    size_t size() const { return elems_.size(); }
    bool empty() const { return elems_.empty(); }

private:

    virtual bool equal(const Type* other) const;
    virtual size_t hash() const;

    anydsl::Array<const Type*> elems_;

    friend class TypeTable;
};

class Generic : public Type {
private:

    Generic(anydsl::Symbol id, const Fct* fct) 
        : id_(id)
        , fct_(fct)
    {}

    virtual bool equal(const Type* t) const;
    virtual size_t hash() const;

    anydsl::Symbol id() const { return id_; }
    const Fct* fct() const { return fct_; }

public:

    virtual void dump(Printer& p) const;
    virtual const anydsl::Type* convert(CodeGen&) const;
    virtual bool is_void() const { return true; }

    anydsl::Symbol id_;
    const Fct* fct_;

    friend class TypeTable;
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
    const NoRet* type_noret() const { return noret_; }
    const Pi* pi(anydsl::ArrayRef<const Type*> elems, const Type* ret);
    const Sigma* sigma(anydsl::ArrayRef<const Type*> elems);
    const Generic* generic(anydsl::Symbol id, const Fct* fct);

    typedef boost::unordered_set<const Type*, TypeHash, TypeEqual> TypeSet;

private:

    TypeSet types_;

    const TypeError* type_error_;
    const Void* type_void_;
    const NoRet* noret_;
#define IMPALA_TYPE(itype, atype) const PrimType* itype##_;
#include "impala/tokenlist.h"
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_TYPE_H
