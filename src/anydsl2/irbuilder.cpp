#include "anydsl2/irbuilder.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

#ifndef NDEBUG
JumpTarget::~JumpTarget() { assert((!lambda_ || lambda_->sealed()) && "JumpTarget not sealed"); }
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
    Lambda* bb = world.basicblock(std::string(name_) + ".crit");
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
    if (cur_bb) {
        jt.jump_from(cur_bb);
        cur_bb = 0;
    }
}

void IRBuilder::branch(const Def* cond, JumpTarget& t, JumpTarget& f) {
    if (cur_bb) {
        if (const PrimLit* lit = cond->isa<PrimLit>())
            jump(lit->box().get_u1().get() ? t : f);
        else {
            cur_bb->branch(cond, t.get(world()), f.get(world()));
            cur_bb = 0;
        }
    }
}

void IRBuilder::call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type) {
    cur_bb = cur_bb->call(to, args, ret_type);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
