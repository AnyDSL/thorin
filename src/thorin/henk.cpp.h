#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

#ifndef HENK_TABLE_TYPE
#error "please define the type table type HENK_TABLE_TYPE"
#endif

#ifndef HENK_STRUCT_EXTRA_NAME
#error "please define the name for HENK_STRUCT_EXTRA_TYPE: HENK_STRUCT_EXTRA_NAME"
#endif

#ifndef HENK_STRUCT_EXTRA_TYPE
#error "please define the type to unify StructTypes HENK_STRUCT_EXTRA_TYPE"
#endif

#ifndef HENK_ENUM_EXTRA_NAME
#error "please define the name for HENK_ENUM_EXTRA_TYPE: HENK_ENUM_EXTRA_NAME"
#endif

#ifndef HENK_ENUM_EXTRA_TYPE
#error "please define the type to unify EnumTypes HENK_ENUM_EXTRA_TYPE"
#endif

#define HENK_UNDERSCORE(N) THORIN_PASTER(N,_)
#define HENK_TABLE_NAME_ HENK_UNDERSCORE(HENK_TABLE_NAME)

size_t Type::gid_counter_ = 1;

Type::Type(HENK_TABLE_TYPE& table, int tag, Types ops)
    : HENK_TABLE_NAME_(&table)
    , tag_(tag)
    , ops_(ops.size())
    , gid_(gid_counter_++)
{
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        if (auto op = ops[i])
            set(i, op);
    }
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t Type::vhash() const {
    if (is_nominal())
        return thorin::murmur3(uint64_t(tag()) << uint64_t(56) | uint64_t(gid()));

    uint64_t seed = thorin::hash_begin(uint8_t(tag()));
    for (auto op : ops_)
        seed = thorin::hash_combine(seed, uint16_t(op->gid()));
    return seed;
}

uint64_t Var::vhash() const {
    return thorin::murmur3(uint64_t(tag()) << uint64_t(56) | uint8_t(depth()));
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Type::equal(const Type* other) const {
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

bool Var::equal(const Type* other) const {
    return other->isa<Var>() ? this->as<Var>()->depth() == other->as<Var>()->depth() : false;
}

//------------------------------------------------------------------------------

/*
 * rebuild
 */

const Type* Type::rebuild(HENK_TABLE_TYPE& to, Types ops) const {
    assert(num_ops() == ops.size());
    if (ops.empty() && &HENK_TABLE_NAME() == &to)
        return this;
    return vrebuild(to, ops);
}

const Type* StructType::vrebuild(HENK_TABLE_TYPE&, Types ops) const {
    assert_unused(this->ops() == ops);
    return this;
}

const Type* EnumType::vrebuild(HENK_TABLE_TYPE&, Types ops) const {
    assert_unused(this->ops() == ops);
    return this;
}

const Type* App      ::vrebuild(HENK_TABLE_TYPE& to, Types ops) const { return to.app(ops[0], ops[1]); }
const Type* TupleType::vrebuild(HENK_TABLE_TYPE& to, Types ops) const { return to.tuple_type(ops); }
const Type* Lambda   ::vrebuild(HENK_TABLE_TYPE& to, Types ops) const { return to.lambda(ops[0], name()); }
const Type* Var      ::vrebuild(HENK_TABLE_TYPE& to, Types    ) const { return to.var(depth()); }

//------------------------------------------------------------------------------

/*
 * reduce
 */

const Type* Type::reduce(int depth, const Type* type, Type2Type& map) const {
    if (auto result = find(map, this))
        return result;
    if (is_monomorphic())
        return this;
    auto new_type = vreduce(depth, type, map);
    return map[this] = new_type;
}

Array<const Type*> Type::reduce_ops(int depth, const Type* type, Type2Type& map) const {
    Array<const Type*> result(num_ops());
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        result[i] = op(i)->reduce(depth, type, map);
    return result;
}

const Type* Lambda::vreduce(int depth, const Type* type, Type2Type& map) const {
    return HENK_TABLE_NAME().lambda(body()->reduce(depth+1, type, map), name());
}

const Type* Var::vreduce(int depth, const Type* type, Type2Type&) const {
    if (this->depth() == depth)
        return type;
    else if (this->depth() > depth)
        return HENK_TABLE_NAME().var(this->depth()-1);  // this is a free variable - shift by one
    else
        return this;                                    // this variable is not free - don't adjust
}

const Type* StructType::vreduce(int depth, const Type* type, Type2Type& map) const {
    auto struct_type = HENK_TABLE_NAME().struct_type(HENK_STRUCT_EXTRA_NAME(), num_ops());
    map[this] = struct_type;
    auto ops = reduce_ops(depth, type, map);

    for (size_t i = 0, e = num_ops(); i != e; ++i)
        struct_type->set(i, ops[i]);

    return struct_type;
}

const Type* EnumType::vreduce(int depth, const Type* type, Type2Type& map) const {
    auto enum_type = HENK_TABLE_NAME().enum_type(HENK_ENUM_EXTRA_NAME(), num_ops());
    map[this] = enum_type;
    auto ops = reduce_ops(depth, type, map);

    for (size_t i = 0, e = num_ops(); i != e; ++i)
        enum_type->set(i, ops[i]);

    return enum_type;
}

const Type* App::vreduce(int depth, const Type* type, Type2Type& map) const {
    auto ops = reduce_ops(depth, type, map);
    return HENK_TABLE_NAME().app(ops[0], ops[1]);
}

const Type* TupleType::vreduce(int depth, const Type* type, Type2Type& map) const {
    return HENK_TABLE_NAME().tuple_type(reduce_ops(depth, type, map));
}

//------------------------------------------------------------------------------

template<class T>
const StructType* TypeTableBase<T>::struct_type(HENK_STRUCT_EXTRA_TYPE HENK_STRUCT_EXTRA_NAME, size_t size) {
    auto type = new StructType(HENK_TABLE_NAME(), HENK_STRUCT_EXTRA_NAME, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

template<class T>
const EnumType* TypeTableBase<T>::enum_type(HENK_ENUM_EXTRA_TYPE HENK_ENUM_EXTRA_NAME, size_t size) {
    auto type = new EnumType(HENK_TABLE_NAME(), HENK_ENUM_EXTRA_NAME, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

template<class T>
const Type* TypeTableBase<T>::app(const Type* callee, const Type* op) {
    auto app = unify(new App(HENK_TABLE_NAME(), callee, op));

    if (auto cache = app->cache_)
        return cache;
    if (auto lambda = app->callee()->template isa<Lambda>()) {
        Type2Type map;
        return app->cache_ = lambda->body()->reduce(1, op, map);
    } else {
        return app->cache_ = app;
    }

    return app;
}

template<class T>
const Type* TypeTableBase<T>::unify_base(const Type* type) {
    auto i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        type = *i;
        return type;
    }

    return insert(type);
}

template<class T>
const Type* TypeTableBase<T>::insert(const Type* type) {
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

template class TypeTableBase<HENK_TABLE_TYPE>;

//------------------------------------------------------------------------------

#undef HENK_STRUCT_EXTRA_NAME
#undef HENK_STRUCT_EXTRA_TYPE
#undef HENK_ENUM_EXTRA_NAME
#undef HENK_ENUM_EXTRA_TYPE
#undef HENK_TABLE_NAME
#undef HENK_TABLE_TYPE
