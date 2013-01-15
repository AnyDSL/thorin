#include "anydsl2/world.h"
#include "verifier.h"

namespace anydsl2 {

class Verifier {
public:
    Verifier(World& world, Lambdas& invalid)
        : world_(world)
        , invalid_(invalid)
        , pass_(world_.new_pass())
    { }

    bool verify() {
        // loop over all lambdas and check them
        for_all(lambda, world_.lambdas()) {
            if(!verify(lambda))
                invalid_.push_back(lambda);
        }
        return invalid_.empty();
    }

private:
    bool verify(const Def* def) {
        if(def->isa<Param>() || def->isa_lambda())
            return true;
        else
            return verify(def->as<PrimOp>());
    }

    bool verify(Lambda* lambda) {
        // check the "body" of this lambda
        for_all(op, lambda->ops()) {
            // -> check the current element for cycles
            if(!verify(op))
                return false;
        }
        return true;
    }

    bool verify(const PrimOp* primop) {
        // if we have detected a cycle -> invalid
        if(primop->is_visited(pass_) && !primop->is_const())
            return false;
        primop->visit(pass_);
        // check all operands recursively
        for_all(op, primop->ops()) {
            if(!verify(op))
                return false;
        }
        primop->unvisit(pass_);
        return true;
    }

private:
    World& world_;
    Lambdas& invalid_;
    size_t pass_;
};

bool verify(World& world, Lambdas& invalid) {
    Verifier ver(world, invalid);
    return ver.verify();
}

}
