#ifndef ANYDSL_JUMP_H
#define ANYDSL_JUMP_H

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/def.h"

namespace anydsl {

class NoRet;

class Jump : public Def {
protected:

    Jump(World& world, const Def* to, const Def* const* begin, const Def* const* end);

public:

    const Def* to() const { return op(0); };
    Args args() const { return Args(*this, 1, numOps()); }
    const NoRet* noret() const;
    std::vector<const Lambda*> succ() const;

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
