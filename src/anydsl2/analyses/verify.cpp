#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"

namespace anydsl2 {

class Verifier {
public:

    Verifier(World& world)
        : world_(world)
        , pass_(world_.new_pass())
    {}

    bool verify();
    bool verify(Lambda* current, Def def, PrimOpSet& primops);
    bool verify_param(Lambda* current, const Param* param);
    bool verify_body(Lambda* lambda);
    bool verify_primop(Lambda* current, const PrimOp* primop, PrimOpSet& primops);
    void invalid(Def def, Def source, const char* msg = nullptr);
    void invalid(Def def, const char* msg) { invalid(def, def, msg); }

    World& world_;
    const size_t pass_;
};

bool Verifier::verify() {
    for (auto lambda : world_.lambdas())
        if (!verify_body(lambda))
            return false;

    return true;
}

bool Verifier::verify_param(Lambda* current, const Param* param) {
    Lambda* plambda = param->lambda();
    const LambdaSet& lambdas = world_.lambdas();
    if (lambdas.find(plambda) == lambdas.end())
        invalid(plambda,  "lambda not contained in the world");
    // is plambda in the history or the current one?
    if (plambda == current)
        return true;
    // param must not be visited in this case
    if (param->is_visited(pass_))
        invalid(param, "invalid cyclic dependencies due to parameters");
    // if the current lambda was already visited
    if (plambda->is_visited(pass_))
        return true;

    // push current lambda into the history and continue with plambda
    current->visit(pass_);

    // we have to mark all params of this lambda as visited
    // since we have to check whether a single one occurs in the history
    for (auto p : current->params())
        p->visit(pass_);

    // continue with the body
    bool result = verify_body(plambda);

    // unmark everything
    current->unvisit(pass_);
    for (auto p : current->params())
        p->unvisit(pass_);

    if (!result)
        invalid(plambda, param);
    return true;
}

bool Verifier::verify(Lambda* current, Def def, PrimOpSet& primops) {
    if (auto param = def->isa<Param>())
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
        invalid(lambda,  "lambda not contained in the world");
    // check the "body" of this lambda
    for (auto op : lambda->ops()) {
        // -> check the current element structure
        if (!verify(lambda, op, primops)) invalid(lambda, op);
    }
    // there are no cycles in this body
    // => thus, we can verfiy types
    const Pi* stype = lambda->type()->isa<Pi>();
    const size_t num_params = lambda->num_params();
    // sanity check: lambda type
    if (!stype || num_params != stype->size())
        invalid(lambda, "invalid structure of parameters and type of the lambda");
    // sanity check: check each parameter type
    for(size_t i = 0; i < num_params; ++i)
        if (!stype->elem(i)->check_with(lambda->param(i)->type()))
            invalid(lambda, "incompatible parameter types");

    // verifiy the actual call
    const Pi* ttype = lambda->to_pi();
    const size_t num_args = lambda->num_args();
    // sanity check: number arguments for call site
    if (!ttype || num_args != ttype->size())
        invalid(lambda, "invalid structure");

    // sanity check: argument types
    for(size_t i = 0; i < num_args; ++i) {
        if (!ttype->elem(i)->check_with(lambda->arg(i)->type()))
            invalid(lambda, "incompatible types");
    }

    // argument types are compatible in general
    // => check inference mapping
    GenericMap generics;
    if (!ttype->infer_with(generics, lambda->arg_pi()))
        invalid(lambda, "type inference not possible");
    // all checks passed => seems to be a valid lambda
    return true;
}

bool Verifier::verify_primop(Lambda* current, const PrimOp* primop, PrimOpSet& primops) {
    // check whether the current primop is stored in the world
    const PrimOpSet& pset = world_.primops();
    if (pset.find(primop) == pset.end())
        invalid(primop,  "primop not contained in the world");

    // if we have detected a cycle -> invalid
    if (primops.find(primop) != primops.end())
        invalid(primop, "invalid cyclic primops");
    primops.insert(primop);

    // check individual primops
    if (auto select = primop->isa<Select>()) {
        if (select->tval()->type() != select->fval()->type())
            invalid(primop, "'select' on unequal types");
        if (select->order() > 0) {
            if (!select->tval()->isa_lambda() || !select->fval()->isa_lambda())
                invalid(select, "higher-order 'select' not on lambda");
            if (select->type() != world_.pi0() && select->type() != world_.pi({world_.mem()}))
                invalid(select, "higher-order 'select' must be of type 'pi()'");
        }
    } else if (auto op = primop->isa<ArithOp>()) {
        if (op->lhs()->type() != op->rhs()->type() || op->type() != op->lhs()->type())
            invalid(op, "'arithop' on unequal types");
        if (!op->type()->isa<PrimType>())
            invalid(op, "'arithop' uses non primitive type");
    } else if (auto op = primop->isa<RelOp>()) {
        if (op->lhs()->type() != op->rhs()->type())
            invalid(op, "'relop' on unequal types");
        if (!op->type()->is_u1())
            invalid(op, "'relop' must yield 'u1'");
    } else if (auto op = primop->isa<TupleOp>()) {
        if (!op->index()->isa<PrimLit>())
            invalid(op, "'tupleop' needs a constant extraction index");
        unsigned index = op->index()->primlit_value<unsigned>();
        const Sigma* tupleType = op->tuple()->type()->isa<Sigma>();
        if (!tupleType)
            invalid(op, "'tupleop' can only work on a tuple");
        if (index >= tupleType->size())
            invalid(op, "'tupleop' index out of bounds");
    } else if (auto op = primop->isa<Store>()) {
        if (auto ptrType = op->ptr()->type()->isa<Ptr>()) {
            if (ptrType->referenced_type() != op->val()->type())
                invalid(op, "ptr must point to the type of the provided value");
        } else
            invalid(op, "ptr requires a pointer type");
    }

    // check all operands recursively
    Def error = 0;
    for (auto op : primop->ops()) {
        if (!verify(current, op, primops))
            error = op;
    }

    primops.erase(primop);

    if (error)
        invalid(primop, error);

    return true;
}

void Verifier::invalid(Def def, Def source, const char* msg) {
    std::cout << "Invalid entry:" << std::endl;
    def->dump();
    if (source != def) {
        std::cout << "caused by " << std::endl;
        source->dump();
    } else if (msg)
        std::cout << msg;

    if (auto lambda = def->isa_lambda())
        lambda->dump_head();
    else
        def->dump();
    assert(false);
}

static void within(World& world, const DefNode* def) {
    if (auto primop = def->isa<PrimOp>())
        assert(world.primops().find(primop) != world.primops().end());
    else if (auto lambda = def->isa_lambda())
        assert(world.lambdas().find(lambda) != world.lambdas().end());
    else
        within(world, def->as<Param>()->lambda());
}

void verify_closedness(World& world) {
    for (auto primop : world.primops()) {
        within(world, primop->representative_);
        for (auto op : primop->ops())
            within(world, op.node());
        for (auto use : primop->uses_)
            within(world, use.def().node());
        for (auto r : primop->representatives_of_)
            within(world, r);
    }
    for (auto lambda : world.lambdas()) {
        assert(lambda->representative_ == lambda && lambda->representatives_of_.empty());
        for (auto op : lambda->ops())
            within(world, op.node());
        for (auto use : lambda->uses_)
            within(world, use.def().node());
        for (auto param : lambda->params()) {
            within(world, param->representative_);
            for (auto use : param->uses_)
                within(world, use.def().node());
            for (auto r : param->representatives_of_)
                within(world, r);
        }
    }
}

//------------------------------------------------------------------------------

void verify(World& world) { return; Verifier(world).verify(); }

//------------------------------------------------------------------------------

} // namespace anydsl2
