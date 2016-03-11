#include "thorin/typetable.h"

namespace thorin {

TypeTable::TypeTable()
    : tuple0_ (unify(new TupleType(*this, Types())))
    , fn0_    (unify(new FnType   (*this, Types(), 0)))
    , mem_    (unify(new MemType  (*this)))
    , frame_  (unify(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) \
    ,T##_(unify(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

const StructAbsType* TypeTable::struct_abs_type(size_t size, size_t num_type_params, const std::string& name) {
    auto struct_abs_type = new StructAbsType(*this, size, num_type_params, name);
    // just put it into the types_ set due to nominal typing
    auto p = types_.insert(struct_abs_type);
    assert_unused(p.second && "hash/equal broken");
    struct_abs_type->hashed_ = true;
    return struct_abs_type;
}

}
