#include "anydsl/memop.h"

#include "anydsl/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

Load::Load(const Def* mem, const Def* ptr)
    : Access(Node_Load, 
             ptr->world().sigma2(ptr->world().mem(), ptr->type()->as<Ptr>()->ref()), 
             2, mem, ptr)
    , extract_mem_(0)
    , extract_val_(0)
{}

const Def* Load::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, 0); 
}

const Def* Load::extract_val() const { 
    return extract_val_ ? extract_val_ : extract_val_ = world().extract(this, 1); 
}

//------------------------------------------------------------------------------

Store::Store(const Def* mem, const Def* ptr, const Def* val)
    : Access(Node_Store, ptr->world().mem(), 3, mem, ptr)
{
    set_op(2, val);
}

//------------------------------------------------------------------------------

Enter::Enter(const Def* mem)
    : MemOp(Node_Enter, mem->world().sigma2(mem->type(), mem->world().frame()), 1, mem)
{}

const Def* Enter::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, 0); 
}

const Def* Enter::extract_frame() const { 
    return extract_frame_ ? extract_frame_ : extract_frame_ = world().extract(this, 1); 
}

//------------------------------------------------------------------------------

Leave::Leave(const Def* mem, const Def* frame)
    : MemOp(Node_Leave, mem->type(), 2, mem)
{
    set_op(1, frame);
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
