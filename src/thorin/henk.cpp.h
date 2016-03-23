#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

#ifndef HENK_TABLE_TYPE
#error "please define the type table type HENK_TABLE_TYPE"
#endif

#ifndef HENK_STRUCT_UNIFIER_NAME
#error "please define the name for HENK_STRUCT_UNIFIER_TYPE: HENK_STRUCT_UNIFIER_NAME"
#endif

size_t Type::gid_counter_ = 1;

//------------------------------------------------------------------------------

const Lambda* close(const Lambda*& lambda, const Type* body) {
    assert(lambda->body() == nullptr);
    const_cast<Lambda*&>(lambda)->set(0, body);

    std::stack<const Type*> stack;
    TypeSet done;

    auto push = [&](const Type* type) {
        if (!type->is_closed() && !done.contains(type)) {
            if (auto de_bruijn = type->isa<DeBruijn>()) {
                if (de_bruijn->lambda() == lambda) {
                    assert(de_bruijn->closed_ == false && de_bruijn->depth_ == -1);
                    de_bruijn->closed_ = true;
                    de_bruijn->depth_  = stack.size();
                }
                done.insert(de_bruijn);
            } else {
                done.insert(type);
                stack.push(type);
                return true;
            }
        }
        return false;
    };

    push(lambda);

    // TODO this is potentially quadratic when closing n types
    while (!stack.empty()) {
        auto type = stack.top();

        bool todo = false;
        for (size_t i = 0, e = type->size(); i != e; ++i)
            todo |= push(type->arg(i));

        if (!todo) {
            stack.pop();
            type->closed_ = true;
            for (size_t i = 0, e = type->size(); i != e && type->closed_; ++i)
                type->closed_ &= type->arg(i)->is_closed();
        }
    }

    for (auto de_bruijn : lambda->de_bruijns())
        assert(de_bruijn->is_closed());

    return lambda->HENK_TABLE_NAME().unify(lambda);
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t Type::vhash() const {
    uint64_t seed = thorin::hash_combine(thorin::hash_begin(int(kind())), size());
    for (auto arg : args_)
        seed = thorin::hash_combine(seed, arg->hash());
    return seed;
}

uint64_t DeBruijn::vhash() const {
    return thorin::hash_combine(thorin::hash_begin(int(kind())), int(lambda()->kind()), depth());
}

uint64_t StructType::vhash() const {
    return thorin::hash_combine(thorin::hash_begin(int(kind())), int(kind()), int(size()), HENK_STRUCT_UNIFIER_NAME());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Type::equal(const Type* other) const {
    bool result = this->kind() == other->kind() && this->size() == other->size()
        && this->is_monomorphic() == other->is_monomorphic();

    if (result) {
        for (size_t i = 0, e = size(); result && i != e; ++i)
            result &= this->arg(i)->is_hashed()
                ? this->arg(i) == other->arg(i)
                : this->arg(i)->equal(other->arg(i));
    }

    return result;
}

bool DeBruijn::equal(const Type* other) const {
    if (auto de_bruijn = other->isa<DeBruijn>())
        return this->lambda()->kind() == lambda()->kind() && this->depth() == de_bruijn->depth();
    return false;
}

bool StructType::equal(const Type* other) const {
    if (auto other_struct_type = other->isa<StructType>())
        return this->HENK_STRUCT_UNIFIER_NAME() == other_struct_type->HENK_STRUCT_UNIFIER_NAME();
    return false;
}

//------------------------------------------------------------------------------

/*
 * rebuild
 */

const Type* Type::rebuild(HENK_TABLE_TYPE& to, Types args) const {
    assert(size() == args.size());
    if (args.empty() && &HENK_TABLE_NAME() == &to)
        return this;
    return vrebuild(to, args);
}

const Type* StructType::vrebuild(HENK_TABLE_TYPE& to, Types args) const {
    auto ntype = to.struct_type(HENK_STRUCT_UNIFIER_NAME(), args.size());
    for (size_t i = 0, e = args.size(); i != e; ++i)
        const_cast<StructType*>(ntype)->set(i, args[i]);
    return ntype;
}

const Type* Application::vrebuild(HENK_TABLE_TYPE& to, Types args) const { return to.application(args[0], args[1]); }
const Type* TupleType  ::vrebuild(HENK_TABLE_TYPE& to, Types args) const { return to.tuple_type(args); }
const Type* DeBruijn   ::vrebuild(HENK_TABLE_TYPE&,    Types     ) const { THORIN_UNREACHABLE; }
const Type* Lambda     ::vrebuild(HENK_TABLE_TYPE&,    Types     ) const { THORIN_UNREACHABLE; }

//------------------------------------------------------------------------------

/*
 * specialize and instantiate
 */

const Type* Lambda::reduce(const Type* type) const {
    Type2Type map;
    for (auto de_bruijn : de_bruijns())
        map[de_bruijn] = type;
    return body()->specialize(map);
}

const Type* Lambda::reduce(Types types) const {
    Type2Type map;
    size_t i = 0;
    const Type* type = this;

    while (auto lambda = type->isa<Lambda>()) {
        auto arg = types[i++];
        for (auto de_bruijn : lambda->de_bruijns()) {
            assert(de_bruijn->is_closed());
            map[de_bruijn] = arg;
        }
        type = lambda->body();
    }

    return type->specialize(map);
}

const Type* Type::specialize(Type2Type& map) const {
    if (auto result = find(map, this))
        return result;
    return vspecialize(map);
}

Array<const Type*> Type::specialize_args(Type2Type& map) const {
    Array<const Type*> result(size());
    for (size_t i = 0, e = size(); i != e; ++i)
        result[i] = arg(i)->specialize(map);
    return result;
}

const Type* Lambda::vspecialize(Type2Type& map) const {
    //auto lambda = HENK_TABLE_NAME().lambda(
    return map[this] = this;
}

const Type* DeBruijn::vspecialize(Type2Type& map) const { return map[this] = this; }

const Type* StructType::vspecialize(Type2Type&) const {
    assert(false && "TODO");
}

const Type* Application::vspecialize(Type2Type& map) const {
    auto args = specialize_args(map);
    return map[this] = HENK_TABLE_NAME().application(args[0], args[1]);
}

const Type* TupleType::vspecialize(Type2Type& map) const {
    return map[this] = HENK_TABLE_NAME().tuple_type(specialize_args(map));
}

//------------------------------------------------------------------------------

template<class T>
const Type* TypeTableBase<T>::application(const Type* callee, const Type* arg) {
    auto app = unify(new Application(HENK_TABLE_NAME(), callee, arg));

    if (app->is_hashed()) {
        if (!app->cache_) {
            if (auto lambda = app->callee()->template isa<Lambda>())
                app->cache_ = lambda->reduce(app->arg());
            else
                app->cache_ = app;
        }
        return app->cache_;
    }

    return app;
}

template<class T>
const Type* TypeTableBase<T>::unify_base(const Type* type) {
    if (type->is_hashed() || !type->is_closed())
        return type;

    auto i = types_.find(type);
    if (i != types_.end()) {
        destroy(type);
        type = *i;
        assert(type->is_hashed());
        return type;
    }

    return insert(type);
}

template<class T>
const Type* TypeTableBase<T>::insert(const Type* type) {
    for (auto arg : type->args()) {
        if (!arg->is_hashed())
            insert(arg);
    }

    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    assert(!type->is_hashed());
    type->hashed_ = true;
    return type;
}

template<class T>
void TypeTableBase<T>::destroy(const Type* type) {
    thorin::HashSet<const Type*> done;
    destroy(type, done);
}

template<class T>
void TypeTableBase<T>::destroy(const Type* type, thorin::HashSet<const Type*>& done) {
    if (!done.contains(type) && !type->is_hashed()) {
        done.insert(type);
        for (auto arg : type->args())
            destroy(arg, done);
        delete type;
    }
}

template class TypeTableBase<HENK_TABLE_TYPE>;

//------------------------------------------------------------------------------

#undef HENK_STRUCT_UNIFIER_NAME
#undef HENK_TABLE_NAME
#undef HENK_TABLE_TYPE
