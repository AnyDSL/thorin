#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

#ifndef HENK_TABLE_NAME_
#error "please define the type table name HENK_TABLE_NAME_"
#endif

#ifndef HENK_NAME_SPACE
#error "please define the namespace via HENK_NAME_SPACE"
#endif

#ifndef HENK_TABLE_TYPE
#error "please define the type table type HENK_TABLE_TYPE"
#endif

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

namespace HENK_NAME_SPACE {

//------------------------------------------------------------------------------

class Type;
class TypeParam;
class HENK_TABLE_TYPE;

template<class T>
struct GIDHash {
    uint64_t operator()(T n) const { return n->gid(); }
};

template<class Key, class Value>
using GIDMap    = HashMap<const Key*, Value, GIDHash<const Key*>>;
template<class Key>
using GIDSet    = HashSet<const Key*, GIDHash<const Key*>>;

template<class To>
using TypeMap      = GIDMap<Type, To>;
using TypeSet      = GIDSet<Type>;
using Type2Type    = TypeMap<const Type*>;

typedef ArrayRef<const Type*> Types;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
class Type : public Streamable, public MagicCast<Type> {
protected:
    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;

    Type(HENK_TABLE_TYPE& HENK_TABLE_NAME, int kind, Types args, size_t num_type_params = 0)
        : HENK_TABLE_NAME_(HENK_TABLE_NAME)
        , kind_(kind)
        , args_(args.size())
        , type_params_(num_type_params)
        , gid_(gid_counter_++)
    {
        for (size_t i = 0, e = num_args(); i != e; ++i) {
            if (auto arg = args[i])
                set(i, arg);
        }
    }

    void set(size_t i, const Type* type) {
        args_[i] = type;
        order_ = std::max(order_, type->order());
        closed_ &= type->is_closed();
        monomorphic_ &= type->is_monomorphic();
    }

public:
    int kind() const { return kind_; }
    HENK_TABLE_TYPE& HENK_TABLE_NAME() const { return HENK_TABLE_NAME_; }

    Types args() const { return args_; }
    const Type* arg(size_t i) const { assert(i < args().size()); return args()[i]; }
    size_t num_args() const { return args_.size(); }
    bool empty() const { return args_.empty(); }

    ArrayRef<const TypeParam*> type_params() const { return type_params_; }
    const TypeParam* type_param(size_t i) const { assert(i < type_params().size()); return type_params()[i]; }
    size_t num_type_params() const { return type_params().size(); }

    bool is_hashed() const { return hashed_; }                ///< This @p Type is already recorded inside of @p HENK_TABLE_TYPE.
    bool is_closed() const { return closed_; }                ///< Are all @p TypeParam%s bound?
    bool is_monomorphic() const { return monomorphic_; }      ///< Does this @p Type not depend on any @p TypeParam%s?.
    bool is_polymorphic() const { return !is_monomorphic(); } ///< Does this @p Type depend on any @p TypeParam%s?.
    int order() const { return order_; }
    size_t gid() const { return gid_; }

    const Type* specialize(Type2Type&) const;
    const Type* instantiate(Type2Type&) const;
    virtual const Type* instantiate(Types) const;
    virtual const Type* vinstantiate(Type2Type&) const = 0;

    const Type* rebuild(HENK_TABLE_TYPE& to, Types args) const;
    const Type* rebuild(Types args) const { return rebuild(HENK_TABLE_NAME(), args); }

    uint64_t hash() const { return is_hashed() ? hash_ : hash_ = vhash(); }
    virtual uint64_t vhash() const;
    virtual bool equal(const Type*) const;

    static size_t gid_counter() { return gid_counter_; }

protected:
    Array<const Type*> specialize_args(Type2Type&) const;

    int order_ = 0;
    mutable uint64_t hash_ = 0;
    mutable bool hashed_ = false;
    mutable bool closed_ = true;
    mutable bool monomorphic_ = true;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const = 0;

    HENK_TABLE_TYPE& HENK_TABLE_NAME_;
    int kind_;
    Array<const Type*> args_;
    mutable Array<const TypeParam*> type_params_;
    mutable size_t gid_;
    static size_t gid_counter_;

    friend const Type* close_base(const Type*&, ArrayRef<const TypeParam*>);
    template<class> friend class TypeTableBase;
};

template<class T>
const T* close(const T*& type, ArrayRef<const TypeParam*> type_param) {
    static_assert(std::is_base_of<Type, T>::value, "T is not a base of thorin::Type");
    return close_base((const Type*&) type, type_param)->template as<T>();
}

class TypeParam : public Type {
private:
    TypeParam(HENK_TABLE_TYPE& HENK_TABLE_NAME, const char* name)
        : Type(HENK_TABLE_NAME, Node_TypeParam, {})
        , name_(name)
    {
        closed_ = false;
        monomorphic_ = false;
    }

public:
    const char* name() const { return name_; }
    const Type* binder() const { return binder_; }
    size_t index() const { return index_; }
    virtual bool equal(const Type*) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    const char* name_;
    mutable const Type* binder_;
    mutable size_t index_;
    mutable const TypeParam* equiv_ = nullptr;

    friend bool Type::equal(const Type*) const;
    friend const Type* close_base(const Type*&, ArrayRef<const TypeParam*>);
    template<class> friend class TypeTableBase;
};

std::ostream& stream_type_params(std::ostream& os, const Type* type);

class TupleType : public Type {
private:
    TupleType(HENK_TABLE_TYPE& HENK_TABLE_NAME, Types args)
        : Type(HENK_TABLE_NAME, Node_TupleType, args)
    {}

    virtual const Type* vinstantiate(Type2Type&) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;

public:
    virtual std::ostream& stream(std::ostream&) const override;

    template<class> friend class TypeTableBase;
};

//------------------------------------------------------------------------------

template<class HENK_TABLE_TYPE>
class TypeTableBase {
private:
    HENK_TABLE_TYPE& HENK_TABLE_NAME() { return *static_cast<HENK_TABLE_TYPE*>(this); }

public:
    struct TypeHash { uint64_t operator () (const Type* t) const { return t->hash(); } };
    struct TypeEqual { bool operator () (const Type* t1, const Type* t2) const { return t1->equal(t2); } };
    typedef HashSet<const Type*, TypeHash, TypeEqual> TypeSet;

    TypeTableBase& operator = (const TypeTableBase&);
    TypeTableBase(const TypeTableBase&);

    TypeTableBase()
        : tuple0_(unify(new TupleType(HENK_TABLE_NAME(), Types())))
    {}
    virtual ~TypeTableBase() { for (auto type : types_) delete type; }

    const TypeParam* type_param(const char* name) { return unify(new TypeParam(HENK_TABLE_NAME(), name)); }
    const TupleType* tuple_type(Types args) { return unify(new TupleType(HENK_TABLE_NAME(), args)); }
    const TupleType* tuple_type() { return tuple0_; } ///< Returns unit, i.e., an empty @p TupleType.

    const TypeSet& types() const { return types_; }

protected:
    const Type* unify_base(const Type* type) {
        if (type->is_hashed() || !type->is_closed())
            return type;

        for (auto& arg : const_cast<Type*>(type)->args_)
            arg = unify_base(arg);

        auto i = types_.find(type);
        if (i != types_.end()) {
            delete type;
            type = *i;
            assert(type->is_hashed());
            return type;
        }

        const auto& p = types_.insert(type);
        assert_unused(p.second && "hash/equal broken");
        assert(!type->is_hashed());
        type->hashed_ = true;
        return type;
    }

    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }

    TypeSet types_;
    const TupleType* tuple0_;///< tuple().

    friend const Type* close_base(const Type*&, ArrayRef<const TypeParam*>);
};

//------------------------------------------------------------------------------

}

#undef HENK_TABLE_NAME
#undef HENK_TABLE_NAME_
#undef HENK_TABLE_TYPE
