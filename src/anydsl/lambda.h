#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <list>
#include <boost/unordered_set.hpp>

#include "anydsl/defuse.h"
#include "anydsl/jump.h"

namespace anydsl {

class Lambda;
class Param;
class Pi;
class Jump;

typedef std::list<Param*> Params;
typedef boost::unordered_set<Param*> ParamSet;
typedef Params::iterator ParamIter;

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

    const Params& params() const { return params_; }

    Jump* jump() { return ops_[0].def()->as<Jump>(); }
    const Jump* jump() const { return ops_[0].def()->as<Jump>(); }

    void insert(Lambda* lambda);
    void remove(Lambda* lambda);

    ParamIter appendParam(const Type* type);

    const Pi* pi() const;

    void setJump(Jump* jump) { assert(!jump_); jump_ = jump; }

private:


    Jump* jump_;
    Params params_;
    ParamSet paramSet_;

    friend class World;
};

} // namespace air

#endif
