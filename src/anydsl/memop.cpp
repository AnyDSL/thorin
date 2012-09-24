#include "anydsl/memop.h"

#include "anydsl/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

Load::Load(const Def* mem, const Def* ptr)
    : Access(   Node_Load, 
                ptr->world().sigma2(ptr->world().mem(), ptr->type()->as<Ptr>()->ref()), 
                2, mem, ptr)
{}

const Def* Load::extract_mem() const {
    return world().extract(this, 0);
}

const Def* Load::extract_val() const {
    return world().extract(this, 1);
}

//------------------------------------------------------------------------------

Store::Store(const Def* mem, const Def* ptr, const Def* val)
    : Access(Node_Store, ptr->world().mem(), 3, mem, ptr)
{
    set_op(2, val);
}

//------------------------------------------------------------------------------

Enter::Enter(const Def* mem)
    : MemOp(Node_Enter, mem->type(), 1, mem)
{}

//------------------------------------------------------------------------------

Leave::Leave(const Def* mem, const Enter* enter)
    : MemOp(Node_Leave, mem->type(), 2, mem)
{
    set_op(1, enter);
}

//------------------------------------------------------------------------------

Slot::Slot(const Enter* enter, const Type* type)
    : PrimOp(Node_Slot, type->world().ptr(type), 1)
{}

bool Slot::equal(const Def* other) const {
    return PrimOp::equal(other) && type() == other->type();
}

size_t Slot::hash() const {
    size_t seed = PrimOp::hash();
    boost::hash_combine(seed, type());
    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl
