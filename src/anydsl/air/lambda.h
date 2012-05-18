#ifndef ANYDSL_AIR_LAMBDA_H
#define ANYDSL_AIR_LAMBDA_H

#include <boost/unordered_set.hpp>

#include "anydsl/air/def.h"

namespace anydsl {

class Lambda;
class Terminator;

typedef boost::unordered_set<Lambda*> Fix;

class Lambda : public Def {
private:

    /**
     * Use this constructor if you know the type beforehand.
     * You are still free to append other params later on.
     */
    Lambda(Lambda* parent, const Type* type);

    /**
     * Use this constructor if you want to incrementally build the type.
     * Initially the type is set to "pi()".
     */
    Lambda(World& world, Lambda* parent);
    ~Lambda();

public:

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

    friend class World;
};


} // namespace air

#endif // ANYDSL_AIR_LAMBDA_H
