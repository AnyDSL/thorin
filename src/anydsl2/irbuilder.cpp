#include "anydsl2/irbuilder.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

#ifndef NDEBUG
JumpTarget::~JumpTarget() { if (lambda_) assert(lambda_->sealed() && "JumpTarget not sealed"); }
#endif

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

void JumpTarget::untangle_first() {
    if (!first_)
        return;
    assert(lambda_);
    Lambda* bb = world().basicblock(name_);
    lambda_->jump0(bb);
    lambda_ = bb;
    first_ = false;
}

void JumpTarget::jump_from(Lambda* bb) {
        if (!lambda_) {
            lambda_ = bb;
            first_ = true;
        } else {
            untangle_first();

            bb->jump0(lambda_);
        }
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
	if (!lambda_)
        return lambda_ = world.basicblock(name_);
    untangle_first();

    return lambda_;
}

//------------------------------------------------------------------------------

void IRBuilder::jump(JumpTarget& jt) {
    if (cur_bb && cur_bb != jt.lambda_) {
        jt.jump_from(cur_bb);

        cur_bb = 0;
    }
}

void IRBuilder::branch(const Def* cond, JumpTarget& t, JumpTarget& f) {
    if (cur_bb) {
        if (const PrimLit* lit = cond->isa<PrimLit>())
            jump(lit->box().get_u1().get() ? t : f);
        else
            cur_bb->branch(cond, t.get(world()), f.get(world()));

        cur_bb = 0;
    }
}

void IRBuilder::call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type) {
    cur_bb = cur_bb->call(to, args, ret_type);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
