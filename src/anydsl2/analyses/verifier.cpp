#include "anydsl2/analyses/verifier.h"

#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/literal.h"
#include "anydsl2/printer.h"

namespace anydsl2 {

class Verifier {
public:

    Verifier(World& world)
        : world_(world)
        , pass_(world_.new_pass())
    {}

    bool verify();
    bool verify(Lambda* current, const Def* def, PrimOpSet& primops);
    bool verify_param(Lambda* current, const Param* param);
    bool verify_body(Lambda* lambda);
    bool verify_primop(Lambda* current, const PrimOp* primop, PrimOpSet& primops);
    bool invalid(const Def* def, const Def* source, const char* msg = 0);
    bool invalid(const Def* def, const char* msg) { return invalid(def, def, msg); }

    World& world_;
    const size_t pass_;
};

bool Verifier::verify() {
    // loop over all lambdas and check them
    bool result = true;
    for_all (lambda, world_.lambdas())
            result &= verify_body(lambda);

    return result;
}

bool Verifier::verify_param(Lambda* current, const Param* param) {
    Lambda* plambda = param->lambda();
    const LambdaSet& lambdas = world_.lambdas();
    if (lambdas.find(plambda) == lambdas.end())
        return invalid(plambda,  "lambda not contained in the world");
    // is plambda in the history or the current one?
    if (plambda == current)
        return true;
    // param must not be visited in this case
    if (param->is_visited(pass_))
        return invalid(param, "invalid cyclic dependencies due to parameters");
    // if the current lambda was already visited
    if (plambda->is_visited(pass_))
        return true;

    // push current lambda into the history and continue with plambda
    current->visit(pass_);

    // we have to mark all params of this lambda as visited
    // since we have to check whether a single one occurs in the history
    for_all (p, current->params())
        p->visit(pass_);

    // continue with the body
    bool result = verify_body(plambda);

    // unmark everything
    current->unvisit(pass_);
    for_all (p, current->params())
        p->unvisit(pass_);

    if (!result)
        return invalid(plambda, param);
    return true;
}

bool Verifier::verify(Lambda* current, const Def* def, PrimOpSet& primops) {
    if (const Param* param = def->isa<Param>())
        return verify_param(current, param);
    else if (def->isa_lambda())
        return true;
    else
        return verify_primop(current, def->as<PrimOp>(), primops);
}

bool Verifier::verify_body(Lambda* lambda) {
    if (lambda->empty()) 
        return true;
    PrimOpSet primops;
    // check whether the lambda is stored in the world
    const LambdaSet& lambdas = world_.lambdas();
    if (lambdas.find(lambda) == lambdas.end())
        return invalid(lambda,  "lambda not contained in the world");
    // check the "body" of this lambda
    for_all (op, lambda->ops()) {
        // -> check the current element structure
        if (!verify(lambda, op, primops)) return invalid(lambda, op);
    }
    // there are no cycles in this body
    // => thus, we can verfiy types
    const Pi* stype = lambda->type()->isa<Pi>();
    const size_t num_params = lambda->num_params();
    // sanity check: lambda type
    if (!stype || num_params != stype->size())
        return invalid(lambda, "invalid structure of parameters and type of the lambda");
    // sanity check: check each parameter type
    for(size_t i = 0; i < num_params; ++i)
        if (!stype->elem(i)->check_with(lambda->param(i)->type()))
            return invalid(lambda, "incompatible parameter types");

    // verifiy the actual call
    const Pi* ttype = lambda->to_pi();
    const size_t num_args = lambda->num_args();
    // sanity check: number arguments for call site
    if (!ttype || num_args != ttype->size())
        return invalid(lambda, "invalid structure");

    // sanity check: argument types
    for(size_t i = 0; i < num_args; ++i) {
        if (!ttype->elem(i)->check_with(lambda->arg(i)->type()))
            return invalid(lambda, "incompatible types");
    }

    // argument types are compatible in general
    // => check inference mapping
    GenericMap generics;
    if (!ttype->infer_with(generics, lambda->arg_pi()))
        return invalid(lambda, "type inference not possible");
    // all checks passed => seems to be a valid lambda
    return true;
}

bool Verifier::verify_primop(Lambda* current, const PrimOp* primop, PrimOpSet& primops) {
    // check whether the current primop is stored in the world
    const PrimOpSet& pset = world_.primops();
    if (pset.find(primop) == pset.end())
        return invalid(primop,  "primop not contained in the world");

    // if we have detected a cycle -> invalid
    if (primops.find(primop) != primops.end())
        return invalid(primop, "invalid cyclic primops");
    primops.insert(primop);

    // check individual primops
    if (const Select* select = primop->isa<Select>()) {
        if (select->tval()->type() != select->fval()->type())
            return invalid(primop, "'select' on unequal types");
        if (select->order() > 0) {
            if (!select->tval()->isa_lambda() || !select->fval()->isa_lambda())
                return invalid(select, "higher-order 'select' not on lambda");
            if (select->type() != world_.pi0() && select->type() != world_.pi1(world_.mem()))
                return invalid(select, "higher-order 'select' must be of type 'pi()'");
        }
    } else if (const ArithOp* op = primop->isa<ArithOp>()) {
        if (op->lhs()->type() != op->rhs()->type() || op->type() != op->lhs()->type())
            return invalid(op, "'arithop' on unequal types");
        if (!op->type()->isa<PrimType>())
            return invalid(op, "'arithop' uses non primitive type");
    } else if (const RelOp* op = primop->isa<RelOp>()) {
        if (op->lhs()->type() != op->rhs()->type())
            return invalid(op, "'relop' on unequal types");
        if (!op->type()->is_u1())
            return invalid(op, "'relop' must yield 'u1'");
    } else if (const TupleOp* op = primop->isa<TupleOp>()) {
        if(!op->index()->isa<PrimLit>())
            return invalid(op, "'tupleop' needs a constant extraction index");
        unsigned index = op->index()->primlit_value<unsigned>();
        const Sigma* tupleType = op->tuple()->type()->isa<Sigma>();
        if(!tupleType)
            return invalid(op, "'tupleop' can only work on a tuple");
        if(index >= tupleType->size())
            return invalid(op, "'tupleop' index out of bounds");
    }

    // check all operands recursively
    const Def* error = 0;
    for_all (op, primop->ops()) {
        if (!verify(current, op, primops)) 
            error = op;
    }

    primops.erase(primop);

    if (error)
        return invalid(primop, error);

    return true;
}

bool Verifier::invalid(const Def* def, const Def* source, const char* msg) {
    std::ostream& o = std::cerr;
    Printer printer(o, true);
    o << "Invalid entry [";
    printer.dump_name(def);
    o << "]: ";
    if (source != def) {
        o << "caused by ";
        printer.dump_name(source);
    } else if (msg)
        o << msg;

    o << std::endl;
    if (Lambda* lambda = def->isa_lambda())
        lambda->dump_body(true, 0, o);
    else
        printer.dump(def);
    o << std::endl;
    return false;
}

//------------------------------------------------------------------------------

bool verify(World& world) { return Verifier(world).verify(); }

//------------------------------------------------------------------------------

} // namespace anydsl2
