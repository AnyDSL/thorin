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

const Def* Load::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, world().literal_u32(0)); 
}

const Def* Load::extract_val() const { 
    return extract_val_ ? extract_val_ : extract_val_ = world().extract(this, world().literal_u32(1)); 
}

//------------------------------------------------------------------------------

const Def* Enter::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, world().literal_u32(0)); 
}

const Def* Enter::extract_frame() const { 
    return extract_frame_ ? extract_frame_ : extract_frame_ = world().extract(this, world().literal_u32(1)); 
}

//------------------------------------------------------------------------------

const Def* CCall::extract_mem() const { 
    return extract_mem_ 
         ? extract_mem_ 
         : extract_mem_ = 
              returns_void() 
            ? this
            : world().extract(this, world().literal_u32(0)); 
}

const Def* CCall::extract_retval() const { 
    assert(!returns_void());
    return extract_retval_ ? extract_retval_ : extract_retval_ = world().extract(this, world().literal_u32(1)); 
}

const Type* CCall::rettype() const {
    assert(!returns_void());
    return type()->as<Sigma>()->elem(1);
}

bool CCall::returns_void() const { return type()->isa<Mem>(); }

//------------------------------------------------------------------------------

} // namespace anydsl2
