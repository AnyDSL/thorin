#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <cstring>
#include <iterator>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/enums.h"
#include "anydsl/util/array.h"
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

typedef Array<PhiOp> PhiOps;

//------------------------------------------------------------------------------

class Use {
public:

    Use() {}
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

typedef boost::unordered_set<Use> Uses;

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

    Def(int kind, const Type* type, size_t numops);
    virtual ~Def();

    void setOp(size_t i, const Def* def) { def->registerUse(i, this); ops_[i] = def; }
    void setType(const Type* type);
    void alloc(size_t size);

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    int kind() const { return kind_; }
    bool isCoreNode() const { return ::anydsl::isCoreNode(kind()); }

    NodeKind node_kind() const { assert(isCoreNode()); return (NodeKind) kind_; }

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    const Uses& uses() const { return uses_; }
    const Type* type() const { return type_; }
    bool isType() const { return !type_; }
    World& world() const;

    typedef ArrayRef<const Def*> Ops;

    Ops ops() const { return Ops(ops_); }
    Ops ops(size_t begin, size_t end) const { return Ops(ops_.slice(begin, end)); }
    const Def* op(size_t i) const { return ops_[i]; }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.size() == 0; }

    /*
     * check for special literals
     */

    bool isPrimLit(int val) const;
    bool isZero() const { return isPrimLit(0); }
    bool isOne() const { return isPrimLit(1); }
    bool isAllSet() const { return isPrimLit(-1); }

    /**
     * Just do what ever you want with this field.
     * Perhaps you want to attach file/line/col information in this field.
     * \p Location provides convenient functionality to achieve this.
     */
    mutable std::string debug;

private:

    int kind_;
    const Type* type_;
    Array<const Def*> ops_;
    mutable Uses uses_;
    mutable bool flag_;

    friend class World;
    friend class DefHash;
    friend class DefEqual;
    friend const Def* const& representitive(const Def* const* ptr);
    friend class Type;
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

    Param(const Type* type, Lambda* parent, size_t index);
    virtual ~Param();

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    const Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    PhiOps phiOps() const;

private:

    virtual void vdump(Printer& printer) const;

    mutable Lambda* lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
