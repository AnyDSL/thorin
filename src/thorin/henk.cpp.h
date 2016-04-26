#ifndef HENK_TABLE_NAME
#error "please define the type table name HENK_TABLE_NAME"
#endif

size_t Type::gid_counter_ = 1;

Type::Type(HENK_TABLE_TYPE& HENK_TABLE_NAME, int kind, Types args, size_t num_type_params)
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

const Type* close_base(const Type*& type, ArrayRef<const TypeParam*> type_params) {
    assert(type->num_type_params() == type_params.size());

    for (size_t i = 0, e = type->num_type_params(); i != e; ++i) {
        assert(!type_params[i]->is_closed());
        type->type_params_[i] = type_params[i];
        type->type_params_[i]->binder_ = type;
        type->type_params_[i]->closed_ = true;
        type->type_params_[i]->index_ = i;
    }

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

    push(type);

    // TODO this is potentially quadratic when closing n types
    while (!stack.empty()) {
        auto type = stack.top();

        bool todo = false;
        for (auto arg : type->args())
            todo |= push(arg);

        if (!todo) {
            stack.pop();
            type->closed_ = true;
            for (size_t i = 0, e = type->num_args(); i != e && type->closed_; ++i)
                type->closed_ &= type->arg(i)->is_closed();
        }
    }

    return type = type->HENK_TABLE_NAME().unify_base(type);
}

uint64_t Type::vhash() const {
    uint64_t seed = thorin::hash_combine(thorin::hash_begin((int) kind()), num_args(), num_type_params());
    for (auto arg : args_)
        seed = thorin::hash_combine(seed, arg->hash());
    return seed;
}

uint64_t TypeParam::vhash() const {
    return thorin::hash_combine(thorin::hash_begin(int(kind())), index(),
                                int(binder()->kind()), binder()->num_type_params(), binder()->num_args());
}

bool Type::equal(const Type* other) const {
    bool result =  this->kind() == other->kind()     &&  this->is_monomorphic() == other->is_monomorphic()
            && this->num_args() == other->num_args() && this->num_type_params() == other->num_type_params();

    if (result) {
        if (is_monomorphic()) {
            for (size_t i = 0, e = num_args(); result && i != e; ++i)
                result &= this->args_[i] == other->args_[i];
        } else {
            for (size_t i = 0, e = num_type_params(); result && i != e; ++i) {
                assert(this->type_param(i)->equiv_ == nullptr);
                this->type_param(i)->equiv_ = other->type_param(i);
            }

            for (size_t i = 0, e = num_args(); result && i != e; ++i)
                result &= this->args_[i]->equal(other->args_[i]);

            for (auto type_param : type_params())
                type_param->equiv_ = nullptr;
        }
        //for (size_t i = 0, e = num_type_params(); result && i != e; ++i) {
            //assert(this->type_param(i)->equiv_ == nullptr);
            //this->type_param(i)->equiv_ = other->type_param(i);
        //}

        //for (size_t i = 0, e = num_args(); result && i != e; ++i)
            //result &= this->arg(i)->is_hashed()
                //? this->arg(i) == other->arg(i)
                //: this->arg(i)->equal(other->arg(i));

        //for (auto type_param : type_params())
            //type_param->equiv_ = nullptr;
    }

    return result;
}

bool TypeParam::equal(const Type* other) const {
    if (auto type_param = other->isa<TypeParam>())
        return this->equiv_ == type_param;
    return false;
}

#undef HENK_TABLE_NAME
