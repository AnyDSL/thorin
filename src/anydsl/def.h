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
typedef ArrayRef<const Def*> Ops;

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

    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

protected:

    Def(int kind, const Type* type, size_t size);
    Def(const Def&);
    virtual ~Def();

    /** 
     * @brief Makes a polymorphic copy.
     *
     * All operands and attributes are copied;
     * all operands register themselves as new uses.
     * The copy itself does not introduce new uses.
     * Most likely, you want to update the newly created node.
     * The return pointer is \em not const.
     * Thus, you are free to run \p update before inserting this node into the \p World again.
     * 
     * @return A modifiable copy of this node.
     */
    virtual Def* clone() const = 0;

    void setOp(size_t i, const Def* def) { def->registerUse(i, this); ops_[i] = def; }
    void setType(const Type* type) { type_ = type; }
    void alloc(size_t size);
    void realloc(size_t size);
    void shrink(size_t newsize) { ops_.shrink(newsize); }

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    void registerUse(size_t i, const Def* def) const;
    void unregisterUse(size_t i, const Def* def) const;

    int kind() const { return kind_; }
    bool is_corenode() const { return ::anydsl::is_corenode(kind()); }

    NodeKind node_kind() const { assert(is_corenode()); return (NodeKind) kind_; }

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    const Uses& uses() const { return uses_; }

    /**
     * Copies all use-info into an array.
     * Useful if you want to modfy users while iterating over all users.
     */
    Array<Use> copy_uses() const;
    const Type* type() const { return type_; }
    bool isType() const { return !type_; }
    World& world() const;

    Ops ops() const { return Ops(ops_); }
    Ops ops(size_t begin, size_t end) const { return Ops(ops_.slice(begin, end)); }
    const Def* op(size_t i) const { return ops_[i]; }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.size() == 0; }

    /// Updates operand indices \p x to point to the corresponding \p ops instead.
    void update(ArrayRef<size_t> idx, ArrayRef<const Def*> ops);

    /// Updates operand \p i to point to \p def instead.
    void update(size_t i, const Def* def) {
        op(i)->unregisterUse(i, this);
        setOp(i, def);
    }

    /*
     * check for special literals
     */

    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }

    /// Just do what ever you want with this field.
    mutable std::string debug;

    union Scratch {
        size_t index;
        int    i;
        void*  ptr;
        bool   marker;

    };

    /** 
     * Use this field in order to annotate information on this Def.
     * Various analyses have to memorize different stuff temporally.
     * Each analysis can use this field for its specific information. 
     * \attention { 
     *      Each pass/analysis simply overwrites this field again.
     *      So keep this in mind and perform copy operations in order to
     *      save your data before running the next pass/analysis.
     *      Also, keep in mind to perform clean up operations at the end 
     *      of your pass/analysis.
     * }
     */
    mutable Scratch scratch;

    /*
     * scratch operations
     */

    void mark() const { scratch.marker = true; }
    void unmark() const { scratch.marker = false; }
    bool is_marked() const { return scratch.marker; }

private:

    int kind_;
    const Type* type_;
    Array<const Def*> ops_;
    mutable Uses uses_;

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

    Param(const Type* type, Lambda* parent, size_t index);
    virtual ~Param();
    virtual Param* clone() const { ANYDSL_UNREACHABLE; }

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    const Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    PhiOps phi() const;

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
