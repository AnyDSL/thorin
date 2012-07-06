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
typedef Def::FilteredUses<Param> UnorderedParams;
typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:

    Lambda();
    Lambda(const Pi* pi);

    bool final() const { return final_; }
    const Jump* jump() const { return op(0)->as<Jump>(); }
    const Pi* pi() const;

    const Param* appendParam(const Type* type);
    void calcType(World& world);

    Callers callers() const { return Callers(uses()); }
    /// Fast but unsorted.
    UnorderedParams unordered_params() const { return UnorderedParams(uses()); }
    /// Slow but sorted.
    Params params() const;
    size_t numParams() const;

    void setJump(const Jump* jump);

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    void dump(bool fancy = false) const;

private:

    virtual void vdump(Printer& printer) const;

    bool final_;
    int numArgs_;

    friend class World;
};

} // namespace anydsl

#endif
