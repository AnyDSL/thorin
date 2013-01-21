#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/printer.h"
#include "verifier.h"

namespace anydsl2 {

#define INVALID(def, msg) \
        return reportInvalidEntry(def, def, msg);

#define INVALID_CE(def, source) \
        return reportInvalidEntry(def, source);

#define VALID return true;

class Verifier {
public:
    Verifier(World& world)
        : world_(world)
        , pass_(world_.new_pass())
    { }

    bool verify() {
        // loop over all lambdas and check them
        bool result = true;
        for_all(lambda, world_.lambdas()) {
            result &= verifyBody(lambda);
        }
        return result;
    }

private:
    bool verify(Lambda* current, const Def* def) {
        if(const Param* param = def->isa<Param>())
            return verifyParam(current, param);
        else if(def->isa_lambda())
            VALID
        else
            return verifyPrimop(current, def->as<PrimOp>());
    }

    bool verifyParam(Lambda* current, const Param* param) {
        Lambda* plambda = param->lambda();

        const LambdaSet& lambdas = world_.lambdas();
        if(lambdas.find(plambda) == lambdas.end())
            INVALID(plambda,  "lambda not contained in the world")

        // is plambda in the history or the current one?
        if(plambda == current)
            VALID
        // param must not be visited in this case
        if(param->is_visited(pass_))
            INVALID(param, "invalid cyclic dependencies")

        if(plambda->is_visited(pass_))
            VALID

        // push current lambda into the history and continue with plambda
        current->visit(pass_);
        // we have to mark all params of this lambda as visited
        // since we have to check whether a single one occurs in the history
        for_all(p, current->params())
            p->visit(pass_);

        // continue with the body
        bool result = verifyBody(plambda);

        // unmark everything
        current->unvisit(pass_);
        for_all(p, current->params())
            p->unvisit(pass_);

        if(!result)
            INVALID_CE(param, plambda)

        VALID
    }

    bool verifyBody(Lambda* lambda) {
        // check whether the lambda is stored in the world
        const LambdaSet& lambdas = world_.lambdas();
        if(lambdas.find(lambda) == lambdas.end())
            INVALID(lambda,  "lambda not contained in the world")
        // check the "body" of this lambda
        for_all(op, lambda->ops()) {
            // -> check the current element structure
            if(!verify(lambda, op)) INVALID_CE(lambda, op)
        }
        // there are no cycles in this body
        // => thus, we can verfiy types
        const Pi* stype = lambda->type()->isa<Pi>();
        const size_t num_params = lambda->num_params();
        // sanity check: lambda type
        if(!stype || num_params != stype->size())
            INVALID(lambda, "invalid structure of parameters and type of the lambda")
        // sanity check: check each parameter type
        for(size_t i = 0; i < num_params; ++i)
            if(!stype->elem(i)->check_with(lambda->param(i)->type()))
                INVALID(lambda, "incompatible parameter types")

        // verifiy the actual call
        const Pi* ttype = lambda->to()->type()->isa<Pi>();
        const size_t num_args = lambda->num_args();
        // sanity check: number arguments for call site
        if(!ttype || num_args != ttype->size())
            INVALID(lambda, "invalid structure")

        // sanity check: argument types
        for(size_t i = 0; i < num_args; ++i) {
            if(!ttype->elem(i)->check_with(lambda->arg(i)->type()))
                INVALID(lambda, "incompatible types")
        }

        // argument types are compatible in general
        // => check inference mapping
        GenericMap generics;
        if(!ttype->infer_with(generics, lambda->arg_pi()))
            INVALID(lambda, "type inference not possible")
        // all checks passed => seems to be a valid lambda
        return true;
    }

    bool verifyPrimop(Lambda* current, const PrimOp* primop) {
        // check whether the current primop is stored in the world
        const PrimOpSet& pset = world_.primops();
        if(pset.find(primop) == pset.end())
            INVALID(primop,  "primop not contained in the world")
        // if we have detected a cycle -> invalid
        if(primop->is_visited(pass_) && !primop->is_const())
            INVALID(primop, "invalid cyclic dependencies")
        primop->visit(pass_);
        // check all operands recursively
        for_all(op, primop->ops()) {
            if(!verify(current, op)) INVALID_CE(primop, op)
        }
        primop->unvisit(pass_);
        // the primop seems to be cycle free
        // => TODO: verify types
        return true;
    }

    bool reportInvalidEntry(const Def* def, const Def* source, const char* msg = 0) {
        std::ostream& o = std::cerr;
        Printer printer(o, true);
        o << "Invalid entry [";
        printer.dump_name(def);
        o << "]: ";
        if(source != def) {
            o << "caused by ";
            printer.dump_name(source);
        } else {
            o << msg;
        }
        o << std::endl;
        if(Lambda* lambda = def->isa_lambda())
            lambda->dump(true, 0, o);
        else
            printer.dump(def);
        o << std::endl;
        return false;
    }

private:
    World& world_;
    size_t pass_;
};

bool verify(World& world) {
    Verifier ver(world);
    return ver.verify();
}

}
