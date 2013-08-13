#ifndef IMPALA_TYPE_H
#define IMPALA_TYPE_H

#include <unordered_set>

#include "anydsl2/node.h"

#include "impala/token.h"

namespace anydsl2 {
    class GenericMap;
    class Type;
    class World;
}

namespace impala {

class FnType;
class NoRet;
class PrimType;
class TupleType;
class Type;
class TypeError;

class TypeTable {
public:
    TypeTable();

    const TypeError* type_error() { return type_error_; }
    const NoRet* noret() { return noret_; }
    const TupleType* type_void() { return void_; }
    const PrimType* primtype(TokenKind kind);
#define IMPALA_TYPE(itype, atype) const PrimType* type_##itype() { return itype##_; }
#include "impala/tokenlist.h"
    const FnType* fntype(const Type*);
    const FnType* fntype(anydsl2::ArrayRef<const Type*> elems = anydsl2::ArrayRef<const Type*>(nullptr, 0));
    const TupleType* tupletype(anydsl2::ArrayRef<const Type*> elems);

private:
    const Type* unify_base(const Type* type);
    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }

    std::unordered_set<const Type*> types_;

#define IMPALA_TYPE(itype, atype) const PrimType* itype##_;
#include "impala/tokenlist.h"
    const TypeError* type_error_;
    const NoRet* noret_;
    const TupleType* void_;
};

class Type : public anydsl2::Node {
protected:
    Type(TokenKind kind, size_t size, const std::string& name)
        : Node((int) kind, size, name)
    {}

public:
    TokenKind kind() const { return (TokenKind) anydsl2::Node::kind(); }
    anydsl2::ArrayRef<const Type*> elems() const { return ops_ref<const Type*>(); }
    const Type* elem(size_t i) const { return elems()[i]; }
    virtual const anydsl2::Type* convert(anydsl2::World&) const = 0;
    bool is_bool() const;
    bool is_int() const;
    bool is_float() const;
    bool is_void() const;
    bool check_with(const Type*) const { return true; } // TODO
    bool infer_with(anydsl2::GenericMap& map, const Type* type) const { return true; } // TODO
    //const Type* specialize(const GenericMap& generic_map) const;
    std::ostream& dump() const;
};

class TypeError : public Type {
private:
    TypeError() 
        : Type(Token::TYPE_error, 0, "<error>")
    {}

    virtual const anydsl2::Type* convert(anydsl2::World&) const { assert(false); }

    friend class TypeTable;
};

class NoRet : public Type {
private:
    NoRet() 
        : Type(Token::TYPE_noret, 0, "!")
    {}

    virtual const anydsl2::Type* convert(anydsl2::World&) const { assert(false); }

    friend class TypeTable;
};

class PrimType : public Type {
private:
    PrimType(TokenKind kind)
        : Type(kind, 0, "<primitive type>")
    {}

public:
    virtual const anydsl2::Type* convert(anydsl2::World&) const;

    friend class TypeTable;
};

class CompoundType : public Type {
protected:
    CompoundType(TokenKind kind, anydsl2::ArrayRef<const Type*> elems, const std::string& name)
        : Type(kind, elems.size(), name)
    {
        size_t i = 0;
        for (auto elem : elems)
            set(i++, elem);
    }
};

class FnType : public CompoundType {
private:
    FnType(TypeTable& typetable, anydsl2::ArrayRef<const Type*> elems)
        : CompoundType(Token::PI, elems, "<function type>")
        , typetable_(typetable)
    {}

public:
    virtual const anydsl2::Type* convert(anydsl2::World&) const;
    const Type* return_type() const;

private:
    TypeTable& typetable_;

    friend class TypeTable;
};

class TupleType : public CompoundType {
private:
    TupleType(anydsl2::ArrayRef<const Type*> elems)
        : CompoundType(Token::SIGMA, elems, "<tuple type>")
    {}

    virtual const anydsl2::Type* convert(anydsl2::World&) const;

    friend class TypeTable;
};

} // namespace impala

#endif
