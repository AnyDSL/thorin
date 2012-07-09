#ifndef ANYDSL_LAMBDA_H
#define ANYDSL_LAMBDA_H

#include <list>

#include "anydsl/def.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Lambda;
class Pi;

typedef Def::FilteredUses<Lambda> Callers;
typedef Def::FilteredUses<Param> UnorderedParams;
typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:

    Lambda();
    Lambda(const Pi* pi);

    bool final() const { return final_; }
    const Pi* pi() const;

    const Param* appendParam(const Type* type);
    void calcType(World& world);

    Callers callers() const { return Callers(uses()); }
    /// Fast but unsorted.
    UnorderedParams unordered_params() const { return UnorderedParams(uses()); }
    /// Slow but sorted.
    Params params() const;
    size_t numParams() const;

    void jumps(const Def* to, const Def* const* begin, const Def* const* end);
    template<size_t N>
    void jumps(const Def* to, const Def* const (&args)[N]) { return jumps(to, args, args + N); }
    void branches(const Def* cond, const Def* tto, const Def* fto);

    std::vector<const Lambda*> succ() const;

    void dump(bool fancy = false) const;

    const Def* todef() const { return op(0); };
    Ops args() const { return ops(1, numOps()); }

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer& printer) const;

    bool final_;
    int numArgs_;

    friend class World;
};

} // namespace anydsl

#endif
