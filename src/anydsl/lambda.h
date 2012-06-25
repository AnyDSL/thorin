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

typedef AutoVector<const Param*> Params;

class Lambda : public Value {
public:

    Lambda();
    Lambda(const Pi* pi);

    bool final() const { return final_; }
    const Params& params() const { return params_; }
    const Jump* jump() const { return ops_[0]->as<Jump>(); }
    const Pi* pi() const;

    Param* appendParam(const Type* type);
    void calcType(World& world);

    void setJump(const Jump* jump);

private:

    bool final_;
    Params params_;

    friend class World;
};

} // namespace air

#endif
