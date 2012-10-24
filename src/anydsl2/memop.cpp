#include "anydsl2/memop.h"

#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

MemOp::MemOp(int kind, size_t size, const Type* type, const Def* mem)
    : PrimOp(kind, size, type)
{
    assert(mem->type()->isa<Mem>());
    assert(size >= 1);
    set_op(0, mem);
}

//------------------------------------------------------------------------------

Load::Load(const Def* mem, const Def* ptr)
    : Access(Node_Load, 2, (const Type*) 0, mem, ptr)
    , extract_mem_(0)
    , extract_val_(0)
{
    set_type(world().sigma2(world().mem(), ptr->type()->as<Ptr>()->ref()));
    assert(ptr->type()->isa<Ptr>() && "must load from pointer");
}

const Def* Load::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, world().literal_u32(0)); 
}

const Def* Load::extract_val() const { 
    return extract_val_ ? extract_val_ : extract_val_ = world().extract(this, world().literal_u32(1)); 
}

//------------------------------------------------------------------------------

Store::Store(const Def* mem, const Def* ptr, const Def* val)
    : Access(Node_Store, 3, ptr->world().mem(), mem, ptr)
{
    set_op(2, val);
}

//------------------------------------------------------------------------------

Enter::Enter(const Def* mem)
    : MemOp(Node_Enter, 1, (const Type*) 0, mem)
{
    set_type(world().sigma2(mem->type(), world().frame()));
}

const Def* Enter::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, world().literal_u32(0)); 
}

const Def* Enter::extract_frame() const { 
    return extract_frame_ ? extract_frame_ : extract_frame_ = world().extract(this, world().literal_u32(1)); 
}

//------------------------------------------------------------------------------

Leave::Leave(const Def* mem, const Def* frame)
    : MemOp(Node_Leave, 2, mem->type(), mem)
{
    set_op(1, frame);
}

//------------------------------------------------------------------------------

Slot::Slot(const Def* frame, const Type* type)
    : PrimOp(Node_Slot, 1, type->to_ptr())
{}

bool Slot::equal(const Node* other) const { return this == other; }
size_t Slot::hash() const { return boost::hash_value(this); }

//------------------------------------------------------------------------------

} // namespace anydsl2
