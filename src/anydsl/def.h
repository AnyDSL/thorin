#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <cstring>
#include <iterator>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/enums.h"
#include "anydsl/util/arrayref.h"
#include "anydsl/util/cast.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Def;
class Lambda;
class Printer;
class Sigma;
class Type;
class World;

//------------------------------------------------------------------------------

class PhiOp {
public:

    PhiOp() {}
    PhiOp(const Def* def, const Lambda* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() { return def_; }
    const Lambda* from() { return from_; }

private:

    const Def* def_;
    const Lambda* from_;
};

typedef std::vector<PhiOp> PhiOps;

//------------------------------------------------------------------------------

class Use {
public:

    Use(size_t index, const Def* def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def* def() const { return def_; }

    bool operator == (const Use& use) const {
        return def() == use.def() && index() == use.index();
    }

    bool operator != (const Use& use) const {
        return def() != use.def() || index() != use.index();
    }

private:

    size_t index_;
    const Def* def_;
};

inline size_t hash_value(const Use& use) { 
    size_t seed = 0;
    boost::hash_combine(seed, use.def());
    boost::hash_combine(seed, use.index());

    return seed;
}

typedef boost::unordered_set<Use> UseSet;

//------------------------------------------------------------------------------

class Def : public MagicCast {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(size_t i, const Def* def) const;
    void unregisterUse(size_t i, const Def* def) const;

protected:

    Def(int kind, const Type* type, size_t numOps)
        : kind_(kind) 
        , type_(type)
        , numOps_(numOps)
        , ops_(numOps ? new const Def*[numOps] : 0)
    {
        std::memset(ops_, 0, sizeof(const Def*) * numOps);
    }

    virtual ~Def();

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    void setOp(size_t i, const Def* def) { def->registerUse(i, this); ops_[i] = def; }
    void delOp(size_t i) const { ops_[i] = 0; }
    void setType(const Type* type) { type_ = type; }
    void alloc(size_t size);

public:

    int kind() const { return kind_; }
    bool isCoreNode() const { return ::anydsl::isCoreNode(kind()); }
    bool isPrimType() const { return ::anydsl::isPrimType(kind()); }
    bool isArithOp()  const { return ::anydsl::isArithOp(kind()); }
    bool isRelOp()    const { return ::anydsl::isRelOp(kind()); }
    bool isConvOp()   const { return ::anydsl::isConvOp(kind()); }
    bool isType() const;

    IndexKind indexKind() const { assert(isCoreNode()); return (IndexKind) kind_; }

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    /**
     * Just do what ever you want with this field.
     * Perhaps you want to attach file/line/col information in this field.
     * \p Location provides convenient functionality to achieve this.
     */
    mutable std::string debug;

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    World& world() const;

    typedef ArrayRef<const Def*> Ops;

    size_t numOps() const { return numOps_; }
    Ops ops() const { return Ops(ops_, numOps_); }
    Ops ops(size_t begin, size_t end) const { assert(end <= numOps_); return Ops(ops_ + begin, end - begin); }
    const Def* op(size_t i) const { anydsl_assert(i < numOps_, "index out of bounds"); return ops_[i]; }

    template<class T> T polyOps() const { 
        return T(ops_, numOps_); 
    }
    template<class T> T polyOps(size_t begin, size_t end) const { 
        assert(end <= numOps_); return T(ops_ + begin, end - begin); 
    }

private:

    int kind_;
    const Type* type_;
    size_t numOps_;
    const Def** ops_;
    mutable UseSet uses_;
    mutable bool flag_;

    friend class World;
    friend class DefHash;
    friend class DefEqual;
};

//------------------------------------------------------------------------------

struct DefHash : std::unary_function<const Def*, size_t> {
    size_t operator () (const Def* v) const { return v->hash(); }
};

struct DefEqual : std::binary_function<const Def*, const Def*, bool> {
    bool operator () (const Def* v1, const Def* v2) const { return v1->equal(v2); }
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(const Type* type, const Lambda* parent, size_t index);

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    const Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    PhiOps phiOps() const;

private:

    virtual void vdump(Printer& printer) const;

    const Lambda* lambda_;
    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
