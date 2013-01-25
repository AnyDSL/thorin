#include "anydsl2/jumptarget.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"

namespace anydsl2 {

#ifndef NDEBUG
JumpTarget::~JumpTarget() { if (lambda_) assert(lambda_->sealed() && "JumpTarget not sealed"); }
#endif

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }
void JumpTarget::seal() { assert(lambda_); lambda_->seal(); }

void JumpTarget::untangle_first() {
    assert(first_ && lambda_);
    Lambda* bb = world().basicblock(name_);
    lambda_->jump0(bb);
    lambda_ = bb;
    first_ = false;
}

Lambda* JumpTarget::new_lambda(World& world) {
    assert(!first_ && !lambda_);
    return lambda_ = world.basicblock(name_);
}

Lambda* JumpTarget::get(World& world) {
    if (!lambda_)
        return new_lambda(world);
    
    Lambda* bb = world.basicblock(std::string(name_) + ".crit");
    bb->seal();
    bb->jump0(lambda_);
    return bb;
}

Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
	if (!lambda_)
        return new_lambda(world);
    if (first_) 
        untangle_first();

    return lambda_;
}

} // namespace anydsl2
