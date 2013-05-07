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

const Def* Load::extract_mem() const { return world().extract(this, world().literal(0u)); }
const Def* Load::extract_val() const { return world().extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

const Def* Enter::extract_mem()   const { return world().extract(this, world().literal(0u)); }
const Def* Enter::extract_frame() const { return world().extract(this, world().literal(1u)); }

//------------------------------------------------------------------------------

const Def* CCall::extract_mem() const { return returns_void() ? this : world().extract(this, world().literal_u32(0)); }
const Def* CCall::extract_retval() const { assert(!returns_void()); return world().extract(this, world().literal_u32(1)); } 
const Type* CCall::rettype() const { assert(!returns_void()); return type()->as<Sigma>()->elem(1); }
bool CCall::returns_void() const { return type()->isa<Mem>() != 0; }

//------------------------------------------------------------------------------

} // namespace anydsl2
