#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <list>

#include "anydsl/def.h"
#include "anydsl/jump.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;
class Jump;

typedef Def::FilteredUses<Jump> Callers;
typedef Def::FilteredUses<Param> Params;

class Lambda : public Def {
public:

    Lambda();
    Lambda(const Pi* pi);

    bool final() const { return final_; }
    //const Params& params() const { return params_; }
    const Jump* jump() const { return ops_[0]->as<Jump>(); }
    const Pi* pi() const;

    const Param* appendParam(const Type* type);
    void calcType(World& world);

    Params params() const { return Params(uses_); }
    Callers callers() const { return Callers(uses_); }

    void setJump(const Jump* jump);

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void dump(Printer& printer, bool descent) const;

private:

    bool final_;
    int numArgs_;

    friend class World;
};

} // namespace air

#endif
