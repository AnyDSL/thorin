#include "anydsl2/jumptarget.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"

namespace anydsl2 {

World& JumpTarget::world() const { assert(lambda_); return lambda_->world(); }

void JumpTarget::target_by(Lambda* from) {
	if (!lambda_) {
		lambda_ = from;
		first_ = true;
		return;
	} else if (first_) {
        Lambda* new_bb = world().basicblock(name_);
        lambda_->jump0(new_bb);
        lambda_ = new_bb;
		first_ = false;
	}

    from->jump0(lambda_);
}


Lambda* JumpTarget::enter() {
    if (lambda_ && !first_)
        lambda_->seal();
    return lambda_;
}

Lambda* JumpTarget::enter_unsealed(World& world) {
	if (!lambda_)
        lambda_ = world.basicblock(name_);
    else if (first_) {
        Lambda* new_bb = world.basicblock(name_);
        lambda_->jump0(new_bb);
        lambda_ = new_bb;
		first_ = false;
    }

    return lambda_;
}

} // namespace anydsl2
