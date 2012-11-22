#include "anydsl2/memop.h"

#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

MemOp::MemOp(int kind, size_t size, const Type* type, const Def* mem, const std::string& name)
    : PrimOp(kind, size, type, name)
{
    assert(mem->type()->isa<Mem>());
    assert(size >= 1);
    set_op(0, mem);
}

//------------------------------------------------------------------------------

Load::Load(const Def* mem, const Def* ptr, const std::string& name)
    : Access(Node_Load, 2, (const Type*) 0, mem, ptr, name)
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

Store::Store(const Def* mem, const Def* ptr, const Def* val, const std::string& name)
    : Access(Node_Store, 3, ptr->world().mem(), mem, ptr, name)
{
    set_op(2, val);
}

//------------------------------------------------------------------------------

Enter::Enter(const Def* mem, const std::string& name)
    : MemOp(Node_Enter, 1, (const Type*) 0, mem, name)
    , extract_mem_(0)
    , extract_frame_(0)
{
    // world is zero -> take the world from the type of the memory
    World& world = mem->world();
    set_type(world.sigma2(mem->type(), world.frame()));
}

const Def* Enter::extract_mem() const { 
    return extract_mem_ ? extract_mem_ : extract_mem_ = world().extract(this, world().literal_u32(0)); 
}

const Def* Enter::extract_frame() const { 
    return extract_frame_ ? extract_frame_ : extract_frame_ = world().extract(this, world().literal_u32(1)); 
}

//------------------------------------------------------------------------------

Leave::Leave(const Def* mem, const Def* frame, const std::string& name)
    : MemOp(Node_Leave, 2, mem->type(), mem, name)
{
    set_op(1, frame);
}

//------------------------------------------------------------------------------

Slot::Slot(const Def* frame, const Type* type, const std::string& name)
    : PrimOp(Node_Slot, 1, type->to_ptr(), name)
{}

bool Slot::equal(const Node* other) const { return this == other; }
size_t Slot::hash() const { return boost::hash_value(this); }

//------------------------------------------------------------------------------

CCall::CCall(const Def* mem, const std::string& callee, 
             ArrayRef<const Def*> args, const Type* rettype, bool vararg, const std::string& name)
    : MemOp(Node_CCall, args.size() + 1, (const Type*) 0, mem, name)
    , callee_(callee)
    , vararg_(vararg)
{
    if (rettype)
        set_type(mem->world().sigma2(mem->type(), rettype));
    else
        set_type(mem->type());

    size_t x = 1;
    for_all (arg, args)
        set_op(x, arg);
}

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

bool CCall::equal(const Node* other) const { 
    if (this->MemOp::equal(other)) {
        const CCall* cother = other->as<CCall>();
        return this->callee() == cother->callee() && this->vararg() == cother->vararg();
    }
    return false;
}

size_t CCall::hash() const { 
    size_t seed = MemOp::hash();
    boost::hash_combine(seed, callee());
    boost::hash_combine(seed, vararg());
    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
