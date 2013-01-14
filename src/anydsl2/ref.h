#ifndef ANYDSL2_REF_H
#define ANYDSL2_REF_H

#include <memory>

#include "anydsl2/util/assert.h"
#include "anydsl2/util/autoptr.h"
#include "anydsl2/util/types.h"

namespace anydsl2 {

class BB;
class Def;
class Ref;
class Type;
class World;

typedef std::auto_ptr<const Ref> RefPtr;

class Ref {
public:

    virtual ~Ref() {}

    virtual const Def* load() const = 0;
    virtual void store(const Def* val) const = 0;
    virtual World& world() const = 0;

    inline static RefPtr create(const Def* def);
    inline static RefPtr create(BB* bb, size_t handle, const Type*, const char* name);
    inline static RefPtr create(RefPtr lref, const Def* index);
};

class RVal : public Ref {
public:

    RVal(const Def* def)
        : def_(def)
    {}

    virtual const Def* load() const { return def_; }
    virtual void store(const Def* val) const { ANYDSL2_UNREACHABLE; }
    virtual World& world() const;

private:

    const Def* def_;
};

class VarRef : public Ref {
public:

    VarRef(BB* bb, size_t handle, const Type* type, const char* name)
        : bb_(bb)
        , handle_(handle)
        , type_(type)
        , name_(name)
    {}

    virtual const Def* load() const;
    virtual void store(const Def* def) const;
    virtual World& world() const;

private:

    BB* bb_;
    size_t handle_;
    const Type* type_;
    const char* name_;
};

class TupleRef : public Ref {
public:

    TupleRef(RefPtr lref, const Def* index)
        : lref_(lref)
        , index_(index)
        , loaded_(0)
    {}

    virtual const Def* load() const;
    virtual void store(const Def* val) const;
    virtual World& world() const;

private:

    RefPtr lref_;
    const Def* index_;

    /// Caches loaded value to prevent quadratic blow up in calls.
    mutable const Def* loaded_;
};

RefPtr Ref::create(const Def* def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create(RefPtr lref, const Def* index) { return RefPtr(new TupleRef(lref, index)); }
RefPtr Ref::create(BB* bb, size_t handle, const Type* type, const char* name) { 
    return RefPtr(new VarRef(bb, handle, type, name)); 
}

} // namespace anydsl2

#endif // ANYDSL2_REF_H
