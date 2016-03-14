#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

size_t Type::gid_counter_ = 1;

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t Type::vhash() const {
    uint64_t seed = thorin::hash_combine(thorin::hash_begin((int) kind()), size());
    for (auto arg : args_)
        seed = thorin::hash_combine(seed, arg->hash());
    return seed;
}

uint64_t TypeParam::vhash() const {
    return thorin::hash_combine(thorin::hash_begin(int(kind())), int(type_abs()->kind()), type_abs()->size());
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

bool TypeParam::equal(const Type* other) const {
    if (auto type_param = other->isa<TypeParam>())
        return this->equiv_ == type_param;
    return false;
}

bool TypeAbs::equal(const Type* other) const {
    if (auto other_type_abs = other->isa<TypeAbs>()) {
        assert(this->type_param_->equiv_ == nullptr);
        this->type_param_->equiv_ = other_type_abs->type_param_;

        bool result = this->body()->equal(other_type_abs->body());

        assert(this->type_param_->equiv_ == other_type_abs->type_param_);
        this->type_param_->equiv_ = nullptr;

        return result;
    }
    return false;
}

//------------------------------------------------------------------------------

/*
 * rebuild
 */

const Type* StructType::vrebuild(World& to, Types args) const {
    auto ntype = to.struct_type(name(), args.size());
    for (size_t i = 0, e = args.size(); i != e; ++i)
        const_cast<StructType*>(ntype)->set(i, args[i]);
    return ntype;
}

//------------------------------------------------------------------------------

template<class T>
const Type* TypeTableBase<T>::unify_base(const Type* type) {
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

template<class T>
const TypeAbs* TypeTableBase<T>::type_abs(const TypeParam*& type_param, const Type*& body) {
    assert(!type_param->is_closed());

    auto type_abs = type_param->type_abs_ = new TypeAbs(HENK_TABLE_NAME(), type_param, body);
    type_param->closed_ = true;

    std::stack<const Type*> stack;
    TypeSet done;

    auto push = [&](const Type* type) {
        if (!type->is_closed() && !done.contains(type) && !type->isa<TypeParam>()) {
            done.insert(type);
            stack.push(type);
            return true;
        }
        return false;
    };

    push(body);

    // TODO this is potentially quadratic when closing n types
    while (!stack.empty()) {
        auto type = stack.top();

        bool todo = false;
        for (auto arg : type->args())
            todo |= push(arg);

        if (!todo) {
            stack.pop();
            type->closed_ = true;
            for (size_t i = 0, e = type->size(); i != e && type->closed_; ++i)
                type->closed_ &= type->arg(i)->is_closed();
        }
    }

    return unify(type_abs);
}

template class TypeTableBase<HENK_TABLE_TYPE>;

//------------------------------------------------------------------------------

#undef HENK_TABLE_NAME
#undef HENK_TABLE_TYPE
