#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <boost/container/flat_set.hpp>
#include <boost/unordered_set.hpp>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef boost::unordered_set<const Lambda*> LambdaSet;

struct ParamLess {
    bool operator () (const Param* p1, const Param* p2) const {
        anydsl_assert(p1->lambda() == p2->lambda(), "params belong to different lambdas"); 
        return p1->index() < p2->index(); 
    }
};

typedef boost::container::flat_set<const Param*, ParamLess> Params;

class Lambda : public Def {
public:

    Lambda(const Pi* pi);
    virtual ~Lambda();

    const Param* appendParam(const Type* type);

    LambdaSet targets() const;
    LambdaSet succ() const;
    LambdaSet callers() const;
    const Params& params() const { return params_; }

    const Def* to() const { return op(0); };
    Ops args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;

    void dump(bool fancy = false) const;

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    Params params_;

    friend class World;
    friend class Param;
};

} // namespace anydsl

#endif
