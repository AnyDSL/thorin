#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <list>

#include "anydsl/defuse.h"
#include "anydsl/jump.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Params;
class Pi;
class Jump;

class Lambda : public Def {
private:

    Lambda(const Pi* pi);

public:

    const Params* params() const { return params_; }
    const Jump* jump() const { return ops_[0].def()->as<Jump>(); }
    const Pi* pi() const;

    void setJump(const Jump* jump);

private:

    anydsl::AutoPtr<const Params> params_;

    friend class World;
};

} // namespace air

#endif
