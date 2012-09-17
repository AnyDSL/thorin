#ifndef ANYDSL_REF_H
#define ANYDSL_REF_H

#include "anydsl/util/assert.h"
#include "anydsl/util/types.h"

namespace anydsl {

class Def;
class Var;
class World;

class Ref {
public:

    virtual ~Ref() {}

    virtual const Def* load() const = 0;
    virtual void store(const Def* val) const = 0;
    virtual World& world() const = 0;
};

class RVal : public Ref {
public:

    RVal(const Def* def)
        : def_(def)
    {}

    virtual const Def* load() const { return def_; }
    virtual void store(const Def* val) const { ANYDSL_UNREACHABLE; }
    virtual World& world() const;

private:

    const Def* def_;
};

class VarRef : public Ref {
public:

    VarRef(Var* var)
        : var_(var)
    {}

    Var* var() const { return var_; }

    virtual const Def* load() const;
    virtual void store(const Def* val) const;
    virtual World& world() const;

private:

    Var* var_;
};

class TupleRef : public Ref {
public:

    TupleRef(const Ref* lref, u32 index)
        : lref_(lref)
        , index_(index)
    {}

    u32 index() const { return index_; }
    const Ref* lref() const { return lref_; }

    virtual const Def* load() const;
    virtual void store(const Def* val) const;
    virtual World& world() const { return lref()->world(); }

private:

    const Ref* lref_;
    u32 index_;
};

} // namespace anydsl

#endif // ANYDSL_REF_H
