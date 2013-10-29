#include "anydsl2/irbuilder.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

VarRef::VarRef(Lambda* bb, size_t handle, const Type* type, const char* name)
    : Ref(type->world())
    , bb_(bb)
    , handle_(handle)
    , type_(type)
    , name_(name)
{}

SlotRef::SlotRef(IRBuilder& builder, const Slot* slot)
    : Ref(builder.world())
    , builder_(builder)
    , slot_(slot)
{}

//------------------------------------------------------------------------------

Def VarRef::load() const { return bb_->get_value(handle_, type_, name_); }
Def AggRef::load() const { return loaded_ ? loaded_ : loaded_ = world().extract(lref_->load(), index_); } 

Def SlotRef::load() const { 
    const Load* load = world().load(builder_.get_mem(), slot_); 
    builder_.set_mem(load->extract_mem());
    return load->extract_val(); 
}

Def AggPtrRef::load() const { 
    auto load = world().load(builder_.get_mem(), world().lea(lref_->load(), index_));
    builder_.set_mem(load->extract_mem());
    return load->extract_val(); 
}

void VarRef::store(Def def) const { bb_->set_value(handle_, def); }
void AggRef::store(Def val) const { lref_->store(world().insert(lref_->load(), index_, val)); }

void SlotRef::store(Def val) const { 
    builder_.set_mem(world().store(builder_.get_mem(), slot_, val)); 
}

void AggPtrRef::store(Def val) const { 
    builder_.set_mem(world().store(builder_.get_mem(), world().lea(lref_->load(), index_), val)); 
}

//------------------------------------------------------------------------------

#ifndef NDEBUG
JumpTarget::~JumpTarget() { assert((!lambda_ || first_ || lambda_->is_sealed()) && "JumpTarget not sealed"); }
#endif

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

Lambda* JumpTarget::untangle() {
    if (!first_)
        return lambda_;
    assert(lambda_);
    Lambda* bb = world().basicblock(name_);
    lambda_->jump(bb, {});
    first_ = false;
    return lambda_ = bb;
}

void JumpTarget::jump_from(Lambda* bb) {
    if (!lambda_) {
        lambda_ = bb;
        first_ = true;
    } else
        bb->jump(untangle(), {});
}

Lambda* JumpTarget::get(World& world) {
    Lambda* bb = world.basicblock(lambda_ ? (name_ + std::string(".crit")) : name_);
    jump_from(bb);
    bb->seal();
    return bb;
}

Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
    return lambda_ ? untangle() : lambda_ = world.basicblock(name_);
}

//------------------------------------------------------------------------------

void IRBuilder::jump(JumpTarget& jt) {
    if (is_reachable()) {
        jt.jump_from(cur_bb);
        set_unreachable();
    }
}

void IRBuilder::branch(Def cond, JumpTarget& t, JumpTarget& f) {
    if (is_reachable()) {
        if (auto lit = cond->isa<PrimLit>())
            jump(lit->value().get_u1().get() ? t : f);
        else if (&t == &f)
            jump(t);
        else {
            cur_bb->branch(cond, t.get(world()), f.get(world()));
            set_unreachable();
        }
    }
}

const Param* IRBuilder::cascading_call(Def to, ArrayRef<Def> args, const Type* ret_type) {
    return is_reachable() ? (cur_bb = cur_bb->call(to, args, ret_type))->param(0) : nullptr;
}

void IRBuilder::mem_call(Def to, ArrayRef<Def> args, const Type* ret_type) {
    if (is_reachable())
        (cur_bb = cur_bb->mem_call(to, args, ret_type));
}

void IRBuilder::tail_call(Def to, ArrayRef<Def> args) {
    if (is_reachable()) {
        cur_bb->jump(to, args);
        set_unreachable();
    }
}

void IRBuilder::param_call(const Param* ret_param, ArrayRef<Def> args) {
    if (is_reachable()) {
        cur_bb->jump(ret_param, args);
        set_unreachable();
    }
}

Def IRBuilder::get_mem() { return cur_bb->get_value(1, world().mem(), "mem"); }
void IRBuilder::set_mem(Def def) { if (is_reachable()) cur_bb->set_value(1, def); }

//------------------------------------------------------------------------------

} // namespace anydsl2
