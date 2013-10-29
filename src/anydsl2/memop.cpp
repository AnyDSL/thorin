#include "anydsl2/memop.h"

#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

MemOp::MemOp(size_t size, NodeKind kind, const Type* type, Def mem, const std::string& name)
    : PrimOp(size, kind, type, name)
{
    assert(mem->type()->isa<Mem>());
    assert(size >= 1);
    set_op(0, mem);
}

//------------------------------------------------------------------------------

Load::Load(Def mem, Def ptr, const std::string& name)
    : Access(2, Node_Load, mem->world().sigma({mem->type(), ptr->type()->as<Ptr>()->referenced_type()}), mem, ptr, name)
{}

Def Load::extract_mem() const { return world().tuple_extract(this, world().literal(0u)); }
Def Load::extract_val() const { return world().tuple_extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

Store::Store(Def mem, Def ptr, Def value, const std::string& name)
    : Access(3, Node_Store, mem->type(), mem, ptr, name)
{
    set_op(2, value);
}

//------------------------------------------------------------------------------

Enter::Enter(Def mem, const std::string& name)
    : MemOp(1, Node_Enter, mem->world().sigma({mem->type(), mem->world().frame()}), mem, name)
{}

Def Enter::extract_mem()   const { return world().tuple_extract(this, world().literal(0u)); }
Def Enter::extract_frame() const { return world().tuple_extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

Leave::Leave(Def mem, Def frame, const std::string& name)
    : MemOp(2, Node_Leave, mem->type(), mem, name)
{
    assert(frame->type()->isa<Frame>());
    set_op(1, frame);
}

//------------------------------------------------------------------------------

Slot::Slot(const Type* type, Def frame, size_t index, const std::string& name)
    : PrimOp(1, Node_Slot, type->world().ptr(type), name)
    , index_(index)
{
    set_op(0, frame);
}

//------------------------------------------------------------------------------

LEA::LEA(Def def, Def index, const std::string& name)
    : PrimOp(2, Node_LEA, nullptr, name)
{
    auto ptr = def->type()->as<Ptr>();
    if (auto sigma = ptr->referenced_type()->isa<Sigma>())
        set_type(ptr->world().ptr(sigma->elem_via_lit(index)));
    else {
        auto array = ptr->referenced_type()->as<ArrayType>();
        set_type(ptr->world().ptr(array->elem_type()));;
    }

    set_op(0, def);
    set_op(1, index);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
