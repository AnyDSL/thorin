#include "thorin/memop.h"

#include "thorin/literal.h"
#include "thorin/world.h"

namespace thorin {

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
    : Access(2, Node_Load, ptr->type()->as<Ptr>()->referenced_type(), mem, ptr, name)
{}

//------------------------------------------------------------------------------

Store::Store(Def mem, Def ptr, Def value, const std::string& name)
    : Access(3, Node_Store, mem->type(), mem, ptr, name)
{
    set_op(2, value);
}

//------------------------------------------------------------------------------

Enter::Enter(Def mem, const std::string& name)
    : MemOp(1, Node_Enter, mem->world().frame(), mem, name)
{}

//------------------------------------------------------------------------------

Leave::Leave(Def mem, Def frame, const std::string& name)
    : MemOp(2, Node_Leave, mem->type(), mem, name)
{
    assert(frame->type()->isa<Frame>());
    set_op(1, frame);
}

//------------------------------------------------------------------------------

} // namespace thorin
