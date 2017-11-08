#ifndef THORIN_UTIL_TYPE_TABLE_H
#define THORIN_UTIL_TYPE_TABLE_H

#include "thorin/util/hash.h"
#include "thorin/util/cast.h"
#include "thorin/util/array.h"
#include "thorin/util/stream.h"

namespace thorin {

//------------------------------------------------------------------------------

template<class T>
struct GIDLt {
    bool operator()(T a, T b) const { return a->gid() < b->gid(); }
};

template<class T>
struct GIDHash {
    static uint64_t hash(T n) { return thorin::murmur3(n->gid()); }
    static bool eq(T a, T b) { return a == b; }
    static T sentinel() { return T(1); }
};

template<class Key, class Value>
using GIDMap = thorin::HashMap<Key, Value, GIDHash<Key>>;
template<class Key>
using GIDSet = thorin::HashSet<Key, GIDHash<Key>>;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
template <class TypeTable>
class TypeBase : public RuntimeCast<TypeBase<TypeTable>>, public Streamable {
protected:
    using Type2Type = GIDMap<const TypeBase*, const TypeBase*>;
    using Types     = ArrayRef<const TypeBase*>;

    TypeBase(const TypeBase&) = delete;
    TypeBase& operator=(const TypeBase&) = delete;

    TypeBase(TypeTable& table, int tag, Types ops);

    void set(size_t i, const TypeBase* type) {
        ops_[i] = type;
        order_       = std::max(order_, type->order());
        monomorphic_ &= type->is_monomorphic();
        if (!is_nominal())
            known_ &= type->is_known();
    }

public:
    int tag() const { return tag_; }
    TypeTable& table() const { return *table_; }

    Types ops() const { return ops_; }
    const TypeBase* op(size_t i) const { return ops()[i]; }
    size_t num_ops() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }

    bool is_nominal() const { return nominal_; }              ///< A nominal @p Type is always different from each other @p Type.
    bool is_known()   const { return known_; }                ///< Does this @p Type depend on any @p UnknownType%s?
    bool is_monomorphic() const { return monomorphic_; }      ///< Does this @p Type not depend on any @p Var%s?.
    bool is_polymorphic() const { return !is_monomorphic(); } ///< Does this @p Type depend on any @p Var%s?.
    int order() const { return order_; }
    size_t gid() const { return gid_; }
    uint64_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
    virtual bool equal(const TypeBase*) const;

    const TypeBase* reduce(int, const TypeBase*, Type2Type&) const;
    const TypeBase* rebuild(TypeTable& to, Types ops) const;
    const TypeBase* rebuild(Types ops) const { return rebuild(table(), ops); }

    static size_t gid_counter() { return gid_counter_; }

protected:
    virtual uint64_t vhash() const;
    virtual const TypeBase* vreduce(int, const TypeBase*, Type2Type&) const;

    mutable uint64_t hash_ = 0;
    int order_ = 0;
    mutable bool known_       = true;
    mutable bool monomorphic_ = true;
    mutable bool nominal_     = false;

private:
    virtual const TypeBase* vrebuild(TypeTable& to, Types ops) const = 0;

    mutable TypeTable* table_;
    int tag_;
    thorin::Array<const TypeBase*> ops_;
    mutable size_t gid_;
    static size_t gid_counter_;

    friend TypeTable;
};

//------------------------------------------------------------------------------

/// Base class for all \p TypeTable%s.
template <class Type>
class TypeTableBase {
public:
    struct TypeHash {
        static uint64_t hash(const Type* t) { return t->hash(); }
        static bool eq(const Type* t1, const Type* t2) { return t2->equal(t1); }
        static const Type* sentinel() { return (const Type*)(1); }
    };

    typedef thorin::HashSet<const Type*, TypeHash> TypeSet;

    TypeTableBase& operator=(const TypeTableBase&) = delete;
    TypeTableBase(const TypeTableBase&) = delete;

    TypeTableBase() {}
    virtual ~TypeTableBase() { for (auto type : types_) delete type; }

    const TypeSet& types() const { return types_; }

protected:
    const Type* unify_base(const Type* type);
    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }
    const Type* insert(const Type*);

    TypeSet types_;
};

//------------------------------------------------------------------------------

template <class TypeTable>
size_t TypeBase<TypeTable>::gid_counter_ = 1;

template <class TypeTable>
TypeBase<TypeTable>::TypeBase(TypeTable& table, int tag, Types ops)
    : table_(&table)
    , tag_(tag)
    , ops_(ops.size())
    , gid_(gid_counter_++)
{
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        if (auto op = ops[i])
            set(i, op);
    }
}

template <class TypeTable>
const TypeBase<TypeTable>* TypeBase<TypeTable>::reduce(int depth, const TypeBase* type, Type2Type& map) const {
    if (auto result = find(map, this))
        return result;
    if (is_monomorphic())
        return this;
    auto new_type = vreduce(depth, type, map);
    return map[this] = new_type;
}

template <class TypeTable>
const TypeBase<TypeTable>* TypeBase<TypeTable>::vreduce(int depth, const TypeBase* type, Type2Type& map) const {
    if (num_ops() > 0) {
        Array<const TypeBase*> result(num_ops());
        for (size_t i = 0, e = num_ops(); i != e; ++i)
            result[i] = op(i)->reduce(depth, type, map);
        return vrebuild(table(), result);
    }
    return this;
}

template <class TypeTable>
const TypeBase<TypeTable>* TypeBase<TypeTable>::rebuild(TypeTable& to, Types ops) const {
    assert(num_ops() == ops.size());
    if (ops.empty() && &table() == &to)
        return this;
    return vrebuild(to, ops);
}

template <class TypeTable>
uint64_t TypeBase<TypeTable>::vhash() const {
    if (is_nominal())
        return thorin::murmur3(uint64_t(tag()) << uint64_t(56) | uint64_t(gid()));

    uint64_t seed = thorin::hash_begin(uint8_t(tag()));
    for (auto op : ops_)
        seed = thorin::hash_combine(seed, uint32_t(op->gid()));
    return seed;
}

template <class TypeTable>
bool TypeBase<TypeTable>::equal(const TypeBase* other) const {
    if (is_nominal())
        return this == other;

    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops()
        && this->is_monomorphic() == other->is_monomorphic();

    if (result) {
        for (size_t i = 0, e = num_ops(); result && i != e; ++i)
            result &= this->op(i) == other->op(i);
    }

    return result;
}

//------------------------------------------------------------------------------

template <class Type>
const Type* TypeTableBase<Type>::unify_base(const Type* type) {
    auto i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        type = *i;
        return type;
    }

    return insert(type);
}

template <class Type>
const Type* TypeTableBase<Type>::insert(const Type* type) {
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

//------------------------------------------------------------------------------

}

#endif // THORIN_UTIL_TYPE_TABLE_H
