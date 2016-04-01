#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

#ifndef HENK_TABLE_TYPE
#error "please define the type table type HENK_TABLE_TYPE"
#endif

#ifndef HENK_STRUCT_UNIFIER_TYPE
#error "please define the type to unify StructTypes HENK_STRUCT_UNIFIER_TYPE"
#endif

#ifndef HENK_STRUCT_UNIFIER_NAME
#error "please define the name for HENK_STRUCT_UNIFIER_TYPE: HENK_STRUCT_UNIFIER_NAME"
#endif

#define HENK_UNDERSCORE(N) N##_
#define HENK_TABLE_NAME_          HENK_UNDERSCORE(HENK_TABLE_NAME)
#define HENK_STRUCT_UNIFIER_NAME_ HENK_UNDERSCORE(HENK_STRUCT_UNIFIER_NAME)

//------------------------------------------------------------------------------

class Type;
class Abstraction;
class Var;
class HENK_TABLE_TYPE;

template<class T>
struct GIDHash {
    uint64_t operator()(T n) const { return n->gid(); }
};

template<class Key, class Value>
using GIDMap    = thorin::HashMap<const Key*, Value, GIDHash<const Key*>>;
template<class Key>
using GIDSet    = thorin::HashSet<const Key*, GIDHash<const Key*>>;

template<class To>
using TypeMap      = GIDMap<Type, To>;
using TypeSet      = GIDSet<Type>;
using Type2Type    = TypeMap<const Type*>;

typedef thorin::ArrayRef<const Type*> Types;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
class Type : public thorin::Streamable, public thorin::MagicCast<Type> {
protected:
    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;

    Type(HENK_TABLE_TYPE& table, int kind, Types args)
        : HENK_TABLE_NAME_(table)
        , kind_(kind)
        , args_(args.size())
        , gid_(gid_counter_++)
    {
        for (size_t i = 0, e = size(); i != e; ++i) {
            if (auto arg = args[i])
                set(i, arg);
        }
    }

    void set(size_t i, const Type* type) {
        args_[i] = type;
        order_       = std::max(order_, type->order());
        closed_      &= type->is_closed();
        monomorphic_ &= type->is_monomorphic();
        known_       &= type->is_known();
    }

public:
    int kind() const { return kind_; }
    HENK_TABLE_TYPE& HENK_TABLE_NAME() const { return HENK_TABLE_NAME_; }

    Types args() const { return args_; }
    const Type* arg(size_t i) const { assert(i < args().size()); return args()[i]; }
    size_t size() const { return args_.size(); }
    bool empty() const { return args_.empty(); }

    bool is_hashed() const { return hashed_; }                ///< This @p Type is already recorded inside of @p HENK_TABLE_TYPE.
    bool is_closed() const { return closed_; }                ///< Are all @p Var%s bound?
    bool is_known()  const { return known_; }                 ///< Deos this @p Type depend on any @p UnknownType%s?
    bool is_monomorphic() const { return monomorphic_; }      ///< Does this @p Type not depend on any @p Var%s?.
    bool is_polymorphic() const { return !is_monomorphic(); } ///< Does this @p Type depend on any @p Var%s?.
    int order() const { return order_; }
    size_t gid() const { return gid_; }
    uint64_t hash() const { return is_hashed() ? hash_ : hash_ = vhash(); }
    virtual bool equal(const Type*) const;

    const Type* reduce(int, const Type*, Type2Type&) const;
    const Type* rebuild(HENK_TABLE_TYPE& to, Types args) const;
    const Type* rebuild(Types args) const { return rebuild(HENK_TABLE_NAME(), args); }

    static size_t gid_counter() { return gid_counter_; }

protected:
    virtual uint64_t vhash() const;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const = 0;
    thorin::Array<const Type*> reduce_args(int, const Type*, Type2Type&) const;

    mutable uint64_t hash_ = 0;
    int order_ = 0;
    mutable bool hashed_      = false;
    mutable bool closed_      = true;
    mutable bool known_       = true;
    mutable bool monomorphic_ = true;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const = 0;

    HENK_TABLE_TYPE& HENK_TABLE_NAME_;
    int kind_;
    thorin::Array<const Type*> args_;
    mutable size_t gid_;
    static size_t gid_counter_;

    //friend const Abstraction* close(const Abstraction*, const Type*);
    template<class> friend class TypeTableBase;
};

class Abstraction : public Type {
protected:
    Abstraction(HENK_TABLE_TYPE& table, int kind, const char* name)
        : Type(table, kind, {nullptr})
        , name_(name)
    {
        closed_ = false;
    }
    Abstraction(HENK_TABLE_TYPE& table, int kind, const Type* body, const char* name)
        : Type(table, kind, {body})
        , name_(name)
    {}

public:
    const char* name() const { return name_; }
    const Type* body() const { return arg(0); }

private:
    const char* name_;

    friend class Var;
};

class Lambda : public Abstraction {
private:
    Lambda(HENK_TABLE_TYPE& table, const char* name)
        : Abstraction(table, Node_Lambda, name)
    {}
    Lambda(HENK_TABLE_TYPE& table, const Type* body, const char* name)
        : Abstraction(table, Node_Lambda, body, name)
    {}

public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    template<class> friend class TypeTableBase;
};

class Pi : public Abstraction {
private:
    Pi(HENK_TABLE_TYPE& table, const char* name)
        : Abstraction(table, Node_Pi, name)
    {}
    Pi(HENK_TABLE_TYPE& table, const Type* body, const char* name)
        : Abstraction(table, Node_Pi, body, name)
    {}

public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    template<class> friend class TypeTableBase;
};

//const Abstraction* close(const Abstraction*, const Type*);

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
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    int depth_;

    template<class> friend class TypeTableBase;
};

class Application : public Type {
private:
    Application(HENK_TABLE_TYPE& table, const Type* callee, const Type* arg)
        : Type(table, Node_Application, {callee, arg})
    {}

public:
    const Type* callee() const { return Type::arg(0); }
    const Type* arg() const { return Type::arg(1); }
    virtual std::ostream& stream(std::ostream&) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

private:
    mutable const Type* cache_ = nullptr;
    template<class> friend class TypeTableBase;
};

class TupleType : public Type {
private:
    TupleType(HENK_TABLE_TYPE& table, Types args)
        : Type(table, Node_TupleType, args)
    {}

    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;

public:
    virtual std::ostream& stream(std::ostream&) const override;

    template<class> friend class TypeTableBase;
};

class StructType : public Type {
private:
    StructType(HENK_TABLE_TYPE& table, HENK_STRUCT_UNIFIER_TYPE HENK_STRUCT_UNIFIER_NAME, size_t size)
        : Type(table, Node_StructType, thorin::Array<const Type*>(size))
        , HENK_STRUCT_UNIFIER_NAME_(HENK_STRUCT_UNIFIER_NAME)
    {}

public:
    HENK_STRUCT_UNIFIER_TYPE HENK_STRUCT_UNIFIER_NAME() const { return HENK_STRUCT_UNIFIER_NAME_; }
    void set(size_t i, const Type* type) const { return const_cast<StructType*>(this)->Type::set(i, type); }

private:
    virtual const Type* vrebuild(HENK_TABLE_TYPE& to, Types args) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;
    virtual uint64_t vhash() const override;
    virtual bool equal(const Type*) const override;
    virtual std::ostream& stream(std::ostream&) const override;

    HENK_STRUCT_UNIFIER_TYPE HENK_STRUCT_UNIFIER_NAME_;

    template<class> friend class TypeTableBase;
};

//------------------------------------------------------------------------------

template<class HENK_TABLE_TYPE>
class TypeTableBase {
private:
    HENK_TABLE_TYPE& HENK_TABLE_NAME() { return *static_cast<HENK_TABLE_TYPE*>(this); }

public:
    struct TypeHash { uint64_t operator()(const Type* t) const { return t->hash(); } };
    struct TypeEqual { bool operator()(const Type* t1, const Type* t2) const { return t2->equal(t1); } };
    typedef thorin::HashSet<const Type*, TypeHash, TypeEqual> TypeSet;

    TypeTableBase& operator=(const TypeTableBase&);
    TypeTableBase(const TypeTableBase&);

    TypeTableBase()
        : unit_(unify(new TupleType(HENK_TABLE_NAME(), Types())))
    {}
    virtual ~TypeTableBase() { for (auto type : types_) delete type; }

    const Var* var(int depth) { return unify(new Var(HENK_TABLE_NAME(), depth)); }
    const Lambda* lambda(const char* name) { return new Lambda(HENK_TABLE_NAME(), name); }
    const Lambda* lambda(const Type* body, const char* name) { return unify(new Lambda(HENK_TABLE_NAME(), body, name)); }
    const Pi* pi(const char* name) { return new Pi(HENK_TABLE_NAME(), name); }
    const Pi* pi(const Type* body, const char* name) { return unify(new Pi(HENK_TABLE_NAME(), body, name)); }
    const Type* application(const Type* callee, const Type* arg);
    const TupleType* tuple_type(Types args) { return unify(new TupleType(HENK_TABLE_NAME(), args)); }
    const TupleType* unit() { return unit_; } ///< Returns unit, i.e., an empty @p TupleType.
    const StructType* struct_type(HENK_STRUCT_UNIFIER_TYPE HENK_STRUCT_UNIFIER_NAME, size_t num_args) {
        return unify(new StructType(HENK_TABLE_NAME(), HENK_STRUCT_UNIFIER_NAME, num_args));
    }

    const TypeSet& types() const { return types_; }

protected:
    const Type* unify_base(const Type* type);
    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }
    const Type* insert(const Type*);
    void destroy(const Type*);
    void destroy(const Type*, thorin::HashSet<const Type*>& done);

    TypeSet types_;
    const TupleType* unit_; ///< tuple().

    //friend const Abstraction* close(const Abstraction*, const Type*);
    friend class Lambda;
};

//------------------------------------------------------------------------------

#undef HENK_STRUCT_UNIFIER_NAME
#undef HENK_STRUCT_UNIFIER_TYPE
#undef HENK_TABLE_NAME
#undef HENK_TABLE_TYPE
