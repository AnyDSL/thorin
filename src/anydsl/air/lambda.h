#ifndef ANYDSL_AIR_LAMBDA_H
#define ANYDSL_AIR_LAMBDA_H

#include <list>
#include <boost/unordered_set.hpp>

#include "anydsl/air/defuse.h"

namespace anydsl {

class Lambda;
class Param;
class Pi;
class Jump;

typedef std::list<Param*> Params;
typedef boost::unordered_set<Param*> ParamSet;
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

    Jump* jump() { return jump_; }
    const Jump* jump() const { return jump_; }

    void insert(Lambda* lambda);
    void remove(Lambda* lambda);

    ParamIter appendParam(const Type* type);

    const Pi* pi() const;

    int depth();

private:

    void setJump(Jump* jump) { assert(!jump_); jump_ = jump; }

    Lambda* parent_;
    Jump* jump_;
    Fix fix_;
    Params params_;
    ParamSet paramSet_;

    friend class World;
};


} // namespace air

#endif // ANYDSL_AIR_LAMBDA_H
