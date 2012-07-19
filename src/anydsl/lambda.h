#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <boost/unordered_set.hpp>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef boost::unordered_set<const Lambda*> LambdaSet;
typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:

    Lambda();
    Lambda(const Pi* pi, Params& params);

    bool final() const { return final_; }
    const Pi* pi() const;

    const Param* appendParam(const Type* type);
    void calcType(World& world, const Params& params);

    LambdaSet to() const;
    LambdaSet succ() const;
    LambdaSet callers() const;
    Params params() const;

    void jumps(const Def* to, const Def* const* begin, const Def* const* end);
    template<size_t N>
    void jumps(const Def* to, const Def* const (&args)[N]) { return jumps(to, args, args + N); }
    void branches(const Def* cond, const Def* tto, const Def* fto);


    void dump(bool fancy = false) const;

    const Def* todef() const { return op(0); };
    Ops args() const { return ops(1, numOps()); }

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    bool final_;
    int numArgs_;

    Params params_;

    friend class World;
};

} // namespace anydsl

#endif
