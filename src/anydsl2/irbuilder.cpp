#include "anydsl2/irbuilder.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

World& RVal::world() const { return load()->world(); }

World& VarRef::world() const { return type_->world(); }
const Def* VarRef::load() const { return bb_->get_value(handle_, type_, name_); }
void VarRef::store(const Def* def) const { bb_->set_value(handle_, def); }

const Def* TupleRef::load() const { 
    if (loaded_)
        return loaded_;

    return loaded_ = world().extract(lref_->load(), index_);
}

void TupleRef::store(const Def* val) const { 
    lref_->store(world().insert(lref_->load(), index_, val)); 
}

World& TupleRef::world() const { 
    return loaded_ ? loaded_->world() : lref_->world(); 
}

//------------------------------------------------------------------------------

#ifndef NDEBUG
JumpTarget::~JumpTarget() { assert((!lambda_ || first_ || lambda_->sealed()) && "JumpTarget not sealed"); }
#endif

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

Lambda* JumpTarget::untangle() {
    if (!first_)
        return lambda_;
    assert(lambda_);
    Lambda* bb = world().basicblock(name_);
    lambda_->jump0(bb);
    first_ = false;
    return lambda_ = bb;
}

void JumpTarget::jump_from(Lambda* bb) {
    if (!lambda_) {
        lambda_ = bb;
        first_ = true;
    } else
        bb->jump0(untangle());
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

void IRBuilder::branch(const Def* cond, JumpTarget& t, JumpTarget& f) {
    if (is_reachable()) {
        if (const PrimLit* lit = cond->isa<PrimLit>())
            jump(lit->box().get_u1().get() ? t : f);
        else {
            cur_bb->branch(cond, t.get(world()), f.get(world()));
            set_unreachable();
        }
    }
}

const Param* IRBuilder::call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type) {
    if (is_reachable())
        return (cur_bb = cur_bb->call(to, args, ret_type))->param(0);
    return 0;
}

void IRBuilder::tail_call(const Def* to, ArrayRef<const Def*> args) {
    if (is_reachable()) {
        cur_bb->jump(to, args);
        set_unreachable();
    }
}

void IRBuilder::return_value(const Param* ret_param, const Def* def) {
    if (is_reachable()) {
        cur_bb->jump1(ret_param, def);
        set_unreachable();
    }
}

void IRBuilder::return_void(const Param* ret_param) {
    if (is_reachable()) {
        cur_bb->jump0(ret_param);
        set_unreachable();
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
