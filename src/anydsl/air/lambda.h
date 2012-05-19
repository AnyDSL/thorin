#ifndef ANYDSL_AIR_LAMBDA_H
#define ANYDSL_AIR_LAMBDA_H

#include <list>
#include <boost/unordered_set.hpp>

#include "anydsl/air/def.h"

namespace anydsl {

class Lambda;
class Param;
class Pi;
class Terminator;

typedef std::list<Param*> Params;
typedef Params::iterator ParamIter;

typedef boost::unordered_set<Lambda*> Fix;

class Lambda : public Def {
private:

    /**
     * Use this constructor if you know the type beforehand.
     * You are still free to append other params later on.
     */
    Lambda(const Pi* pi);

    /**
     * Use this constructor if you want to incrementally build the type.
     * Initially the type is set to "pi()".
     */
    Lambda(World& world);
    ~Lambda();

public:

    const Fix& fix() const { return fix_; }
    const Fix& siblings() const { assert(parent_); return parent_->fix(); }
    const Params& params() const { return params_; }

    Terminator* terminator() { return terminator_; }
    const Terminator* terminator() const { return terminator_; }

    void insert(Lambda* lambda);
    void remove(Lambda* lambda);

    ParamIter appendParam(const Type* type);

    const Pi* pi() const;

private:

    void setTerminator(Terminator* terminator) { assert(!terminator_); terminator_ = terminator; }

    Lambda* parent_;
    Terminator* terminator_;
    Fix fix_;
    Params params_;

    friend class World;
};


} // namespace air

#endif // ANYDSL_AIR_LAMBDA_H
