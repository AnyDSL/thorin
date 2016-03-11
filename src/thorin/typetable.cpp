#include "thorin/typetable.h"

namespace thorin {

TypeTable::TypeTable()
    : tuple0_ (unify(new TupleType(*this, Types())))
    , fn0_    (unify(new FnType   (*this, Types(), 0)))
    , mem_    (unify(new MemType  (*this)))
    , frame_  (unify(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) ,T##_(unify(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

TypeTable::~TypeTable() {
    for (auto type   : types_)
        delete type;
}

const StructAbsType* TypeTable::struct_abs_type(size_t size, size_t num_type_params, const std::string& name) {
    auto struct_abs_type = new StructAbsType(*this, size, num_type_params, name);
    // just put it into the types_ set due to nominal typing
    auto p = types_.insert(struct_abs_type);
    assert_unused(p.second && "hash/equal broken");
    struct_abs_type->hashed_ = true;
    return struct_abs_type;
}

const Type* TypeTable::unify_base(const Type* type) {
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

}
