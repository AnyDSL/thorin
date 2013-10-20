#include "anydsl2/memop.h"

#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

MemOp::MemOp(size_t size, int kind, const Type* type, const Def* mem, const std::string& name)
    : PrimOp(size, kind, type, name)
{
    assert(mem->type()->isa<Mem>());
    assert(size >= 1);
    set_op(0, mem);
}

//------------------------------------------------------------------------------

Load::Load(const Def* mem, const Def* ptr, const std::string& name)
    : Access(2, Node_Load, mem->world().sigma({mem->type(), ptr->type()->as<Ptr>()->referenced_type()}), mem, ptr, name)
{}

const Def* Load::extract_mem() const { return world().tuple_extract(this, world().literal(0u)); }
const Def* Load::extract_val() const { return world().tuple_extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

Store::Store(const Def* mem, const Def* ptr, const Def* value, const std::string& name)
    : Access(3, Node_Store, mem->type(), mem, ptr, name)
{
    set_op(2, value);
}

//------------------------------------------------------------------------------

Enter::Enter(const Def* mem, const std::string& name)
    : MemOp(1, Node_Enter, mem->world().sigma({mem->type(), mem->world().frame()}), mem, name)
{}

const Def* Enter::extract_mem()   const { return world().tuple_extract(this, world().literal(0u)); }
const Def* Enter::extract_frame() const { return world().tuple_extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

Leave::Leave(const Def* mem, const Def* frame, const std::string& name)
    : MemOp(2, Node_Leave, mem->type(), mem, name)
{
    assert(frame->type()->isa<Frame>());
    set_op(1, frame);
}

//------------------------------------------------------------------------------

Slot::Slot(const Type* type, const Def* frame, size_t index, const std::string& name)
    : PrimOp(1, Node_Slot, type->world().ptr(type), name)
    , index_(index)
{
    set_op(0, frame);
}

//------------------------------------------------------------------------------

LEA::LEA(const Def* ptr, const Def* index, const std::string& name)
    : PrimOp(2, Node_LEA, ptr->type()->isa<Ptr>() ? ptr->type()->as<Ptr>() : ptr->world().ptr(ptr->type()->as<Sigma>()->elem_via_lit(index)), name)
{
    set_op(0, ptr);
    set_op(1, index);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
