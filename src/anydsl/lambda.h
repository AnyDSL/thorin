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

    Lambda(const Pi* pi);

    const Param* appendParam(const Type* type);

    LambdaSet targets() const;
    LambdaSet succ() const;
    LambdaSet callers() const;
    Params params() const;

    const Def* to() const { return op(0); };
    Ops args() const { return ops().slice_back(1); }
    const Pi* pi() const;

    void dump(bool fancy = false) const;

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    int numParams_;

    Params params_;

    friend class World;
};

} // namespace anydsl

#endif
