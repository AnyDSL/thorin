#include "anydsl2/jumptarget.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"

namespace anydsl2 {

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

void JumpTarget::untangle_first() {
    assert(first_ && lambda_);
    Lambda* new_lambda = world().basicblock(name_);
    lambda_->jump0(new_lambda);
    lambda_ = new_lambda;
    first_ = false;
}

void JumpTarget::new_lambda(World& world) {
    assert(!first_ && !lambda_);
    lambda_ = world.basicblock(name_);
}

Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
	if (!lambda_)
        new_lambda(world);
    else if (first_) 
        untangle_first();

    return lambda_;
}

void JumpTarget::jump(JumpTarget& to) {
    if (lambda_)
        lambda_->jump(to);
}

void JumpTarget::branch(const Def* cond, JumpTarget& tto, JumpTarget& fto) {
    if (lambda_)
        lambda_->branch(cond, tto, fto);
}

} // namespace anydsl2
