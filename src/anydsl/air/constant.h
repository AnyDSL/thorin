#ifndef ANYDSL_AIR_CONSTANT_H
#define ANYDSL_AIR_CONSTANT_H

#include <vector>

#include <boost/unordered_set.hpp>

#include "anydsl/air/def.h"
#include "anydsl/support/world.h"
#include "anydsl/util/box.h"

namespace anydsl {

class Type;
class Terminator;

typedef boost::unordered_set<Lambda*> Fix;

//------------------------------------------------------------------------------

class Constant : public Value {
protected:

    Constant(IndexKind index, const Type* type, const std::string& debug)
        : Value(index, type, debug)
    {}
};

typedef std::vector<Constant*> ConstList;

//------------------------------------------------------------------------------

class Undef : public Constant {
public:

    Undef(const Type* type, const std::string& debug = "")
        : Constant(Index_Undef, type, debug)
        {}
};

//------------------------------------------------------------------------------

class PrimConst : public Constant {
public:

    PrimConst(World& world, PrimTypeKind kind, Box box, const std::string& debug = "");

    PrimTypeKind primTypeKind() { return (PrimTypeKind) index(); }
    Box box() const { return box_; }

    virtual uint64_t hash() const;

private:

    Box box_;
};

//------------------------------------------------------------------------------

class Tuple : public Constant {
public:

    const ConstList& consts() const { return consts_; }

    virtual uint64_t hash() const;

private:

    ConstList consts_;
};

//------------------------------------------------------------------------------

class Lambda : public Constant {
public:

    /**
     * Use this constructor if you know the type beforehand.
     * You are still free to append other params later on.
     */
    Lambda(Lambda* parent, const Pi* type, const std::string& debug = "");

    /**
     * Use this constructor if you want to incrementally build the type.
     * Initially the type is set to "pi()".
     */
    Lambda(World& world, Lambda* parent, const std::string& debug = "");
    ~Lambda();

    const Fix& fix() const { return fix_; }
    const Fix& siblings() const { assert(parent_); return parent_->fix(); }

    Terminator* terminator() { return terminator_; }
    const Terminator* terminator() const { return terminator_; }

    void setTerminator(Terminator* Terminator) { terminator_ = Terminator; }

    void insert(Lambda* lambda);
    void remove(Lambda* lambda);

    virtual uint64_t hash() const { return 0; /* TODO */ }

private:

    Lambda* parent_;
    Terminator* terminator_;
    Fix fix_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_CONSTANT_H
