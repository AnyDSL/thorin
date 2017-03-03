#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

#ifndef HENK_TABLE_TYPE
#error "please define the type table type HENK_TABLE_TYPE"
#endif

#ifndef HENK_STRUCT_EXTRA_TYPE
#error "please define the type to unify StructTypes HENK_STRUCT_EXTRA_TYPE"
#endif

#ifndef HENK_STRUCT_EXTRA_NAME
#error "please define the name for HENK_STRUCT_EXTRA_TYPE: HENK_STRUCT_EXTRA_NAME"
#endif

#define HENK_UNDERSCORE(N) THORIN_PASTER(N,_)
#define HENK_TABLE_NAME_ HENK_UNDERSCORE(HENK_TABLE_NAME)
#define HENK_STRUCT_EXTRA_NAME_ HENK_UNDERSCORE(HENK_STRUCT_EXTRA_NAME)

//------------------------------------------------------------------------------

class Type;
class Var;
class HENK_TABLE_TYPE;

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

template<class To>
using TypeMap   = GIDMap<const Type*, To>;
using TypeSet   = GIDSet<const Type*>;
using Type2Type = TypeMap<const Type*>;

typedef thorin::ArrayRef<const Type*> Types;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
class Type : public thorin::MagicCast<Type>, public thorin::Streamable {
protected:
    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;

    Type(HENK_TABLE_TYPE& table, int tag, Types ops);

    void set(size_t i, const Type* type) {
        ops_[i] = type;
        order_       = std::max(order_, type->order());
        monomorphic_ &= type->is_monomorphic();
        if (!is_nominal())
            known_ &= type->is_known();
    }

public:
    int tag() const { return tag_; }
    HENK_TABLE_TYPE& HENK_TABLE_NAME() const { return *HENK_TABLE_NAME_; }

    Types ops() const { return ops_; }
    const Type* op(size_t i) const { return ops()[i]; }
    size_t num_ops() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }

    bool is_nominal() const { return nominal_; }              ///< A nominal @p Type is always different from each other @p Type.
    bool is_known()   const { return known_; }                ///< Does this @p Type depend on any @p UnknownType%s?
    bool is_monomorphic() const { return monomorphic_; }      ///< Does this @p Type not depend on any @p Var%s?.
    bool is_polymorphic() const { return !is_monomorphic(); } ///< Does this @p Type depend on any @p Var%s?.
    int order() const { return order_; }
    size_t gid() const { return gid_; }
    uint64_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
    virtual bool equal(const Type*) const;

    const Type* reduce(int, const Type*, Type2Type&) const;
    const Type* rebuild(HENK_TABLE_TYPE& to, Types ops) const;
    const Type* rebuild(Types ops) const { return rebuild(HENK_TABLE_NAME(), ops); }

    static size_t gid_counter() { return gid_counter_; }

protected:
    virtual uint64_t vhash() const;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const = 0;
    thorin::Array<const Type*> reduce_ops(int, const Type*, Type2Type&) const;

    mutable uint64_t hash_ = 0;
    int order_ = 0;
    mutable bool known_       = true;
    mutable bool monomorphic_ = true;
    mutable bool nominal_     = false;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const = 0;

    mutable HENK_TABLE_TYPE* HENK_TABLE_NAME_;
    int tag_;
    thorin::Array<const Type*> ops_;
    mutable size_t gid_;
    static size_t gid_counter_;

    template<class> friend class TypeTableBase;
};

class Lambda : public Type {
private:
    Lambda(HENK_TABLE_TYPE& table, const Type* body, const char* name)
        : Type(table, Node_Lambda, {body})
        , name_(name)
    {}

public:
    const char* name() const { return name_; }
    const Type* body() const { return op(0); }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    const char* name_;

    template<class> friend class TypeTableBase;
};

class Var : public Type {
private:
    Var(HENK_TABLE_TYPE& table, int depth)
        : Type(table, Node_Var, {})
        , depth_(depth)
    {
        monomorphic_ = false;
    }

public:
    int depth() const { return depth_; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const Type*) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    int depth_;

    template<class> friend class TypeTableBase;
};

class App : public Type {
private:
    App(HENK_TABLE_TYPE& table, const Type* callee, const Type* arg)
        : Type(table, Node_App, {callee, arg})
    {}

public:
    const Type* callee() const { return Type::op(0); }
    const Type* arg() const { return Type::op(1); }
    virtual std::ostream& stream(std::ostream&) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

private:
    mutable const Type* cache_ = nullptr;
    template<class> friend class TypeTableBase;
};

class TupleType : public Type {
private:
    TupleType(HENK_TABLE_TYPE& table, Types ops)
        : Type(table, Node_TupleType, ops)
    {}

    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const override;

public:
    virtual std::ostream& stream(std::ostream&) const override;

    template<class> friend class TypeTableBase;
};

class StructType : public Type {
private:
    StructType(HENK_TABLE_TYPE& table, HENK_STRUCT_EXTRA_TYPE HENK_STRUCT_EXTRA_NAME, size_t size)
        : Type(table, Node_StructType, thorin::Array<const Type*>(size))
        , HENK_STRUCT_EXTRA_NAME_(HENK_STRUCT_EXTRA_NAME)
    {
        nominal_ = true;
    }

public:
    HENK_STRUCT_EXTRA_TYPE HENK_STRUCT_EXTRA_NAME() const { return HENK_STRUCT_EXTRA_NAME_; }
    void set(size_t i, const Type* type) const { return const_cast<StructType*>(this)->Type::set(i, type); }

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;
    virtual std::ostream& stream(std::ostream&) const override;

    HENK_STRUCT_EXTRA_TYPE HENK_STRUCT_EXTRA_NAME_;

    template<class> friend class TypeTableBase;
};

//------------------------------------------------------------------------------

template<class HENK_TABLE_TYPE>
class TypeTableBase {
private:
    HENK_TABLE_TYPE& HENK_TABLE_NAME() { return *static_cast<HENK_TABLE_TYPE*>(this); }

public:
    struct TypeHash {
        static uint64_t hash(const Type* t) { return t->hash(); }
        static bool eq(const Type* t1, const Type* t2) { return t2->equal(t1); }
        static const Type* sentinel() { return (const Type*)(1); }
    };

    typedef thorin::HashSet<const Type*, TypeHash> TypeSet;

    TypeTableBase& operator=(const TypeTableBase&) = delete;
    TypeTableBase(const TypeTableBase&) = delete;

    TypeTableBase()
        : unit_(unify(new TupleType(HENK_TABLE_NAME(), Types())))
    {}
    virtual ~TypeTableBase() { for (auto type : types_) delete type; }

    const Var* var(int depth) { return unify(new Var(HENK_TABLE_NAME(), depth)); }
    const Lambda* lambda(const Type* body, const char* name) { return unify(new Lambda(HENK_TABLE_NAME(), body, name)); }
    const Type* app(const Type* callee, const Type* arg);
    const TupleType* tuple_type(Types ops) { return unify(new TupleType(HENK_TABLE_NAME(), ops)); }
    const TupleType* unit() { return unit_; } ///< Returns unit, i.e., an empty @p TupleType.
    const StructType* struct_type(HENK_STRUCT_EXTRA_TYPE HENK_STRUCT_EXTRA_NAME, size_t size);
    const TypeSet& types() const { return types_; }

    friend void swap(TypeTableBase& t1, TypeTableBase& t2) {
        using std::swap;
        swap(t1.types_, t2.types_);
        swap(t1.unit_,  t2.unit_);

        t1.fix();
        t2.fix();
    }

private:
    void fix() {
        for (auto type : types_)
            type->HENK_TABLE_NAME_ = &HENK_TABLE_NAME();
    }

protected:
    const Type* unify_base(const Type* type);
    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }
    const Type* insert(const Type*);

    TypeSet types_;
    const TupleType* unit_; ///< tuple().

    friend class Lambda;
};

//------------------------------------------------------------------------------

#undef HENK_STRUCT_EXTRA_NAME
#undef HENK_STRUCT_EXTRA_TYPE
#undef HENK_TABLE_NAME
#undef HENK_TABLE_TYPE
