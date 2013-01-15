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
        // reset visit information
        for_all(lambda, world_.lambdas())
            lambda->visit(pass_);

        // loop over all lambdas and check them
        for_all(lambda, world_.lambdas()) {
            // update the current pass
            pass_ = world_.new_pass();
            if(!verify(lambda))
                invalid_.push_back(lambda);
        }
        return invalid_.empty();
    }

private:
    bool visit(const Def* def) {
        if(def->is_visited(pass_))
            return false;
        def->visit(pass_);
        return true;
    }

    bool verify(const Def* def) {
        if(def->isa<Param>())
            return true;
        if(Lambda* lambda = def->isa_lambda()) {
            // already visited lambdas are fine per definition
            if(lambda->cur_pass() <= pass_)
                return true;
            return verify(lambda);
        }
        else
            return verify(def->as<PrimOp>());
    }

    bool verify(Lambda* lambda) {
        // check the "body" of this lambda
        for_all(op, lambda->ops()) {
            if(!verify(op))
                return false;
        }
        return true;
    }

    bool verify(const PrimOp* primop) {
        // if we have detected a cycle -> invalid
        if(!visit(primop) && !primop->is_const())
            return false;
        // check all operands recursively
        for_all(op, primop->ops()) {
            if(!verify(op))
                return false;
        }
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
