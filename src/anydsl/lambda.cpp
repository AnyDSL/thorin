#include "anydsl/lambda.h"

#include <algorithm>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/analyses/scope.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(size_t gid, const Pi* pi, uint32_t flags)
    : Def(Node_Lambda, pi)
    , gid_(gid)
    , flags_(flags)
{
    params_.reserve(pi->size());
}

Lambda::~Lambda() {
    for_all (param, params())
        delete param;
}

Lambda* Lambda::stub() const { 
    Lambda* result = world().lambda(pi(), flags());
    result->debug = debug;

    for (size_t i = 0, e = params().size(); i != e; ++i)
        result->param(i)->debug = param(i)->debug;

    return result;
}

const Pi* Lambda::pi() const { return type()->as<Pi>(); }
const Pi* Lambda::to_pi() const { return to()->type()->as<Pi>(); }

const Param* Lambda::append_param(const Type* type) {
    size_t size = pi()->elems().size();

    Array<const Type*> elems(size + 1);
    *std::copy(pi()->elems().begin(), pi()->elems().end(), elems.begin()) = type;
    set_type(world().pi(elems));

    const Param* param = new Param(type, this, size);
    params_.push_back(param);

    return param;
}

bool Lambda::equal(const Node* other) const { return this == other; }
size_t Lambda::hash() const { return boost::hash_value(this); }

static void find_lambdas(const Def* def, LambdaSet& result) {
    if (Lambda* lambda = def->isa_lambda()) {
        result.insert(lambda);
        return;
    }

    for_all (op, def->ops())
        find_lambdas(op, result);
}

static void find_preds(const Def* def, LambdaSet& result) {
    if (Lambda* lambda = def->isa_lambda())
        result.insert(lambda);
    else {
        anydsl_assert(def->isa<PrimOp>(), "not a PrimOp");

        for_all (use, def->uses())
            find_preds(use.def(), result);
    }
}

LambdaSet Lambda::preds() const {
    LambdaSet result;

    for_all (use, uses())
        find_preds(use.def(), result);

    return result;
}

LambdaSet Lambda::targets() const {
    LambdaSet result;
    find_lambdas(to(), result);

    return result;
}

LambdaSet Lambda::hos() const {
    LambdaSet result;
    for_all (def, args())
        find_lambdas(def, result);

    return result;
}

LambdaSet Lambda::succs() const {
    LambdaSet result;
    for_all (def, ops())
        find_lambdas(def, result);

    return result;
}

template<bool fo>
Array<const Param*> Lambda::classify_params() const {
    Array<const Param*> res(params().size());

    size_t size = 0;
    for_all (param, params())
        if (fo ^ (param->type()->isa<Pi>() != 0))
            res[size++] = param;

    res.shrink(size);

    return res;
}

template<bool fo>
Array<const Def*> Lambda::classify_args() const {
    Array<const Def*> res(args().size());

    size_t size = 0;
    for_all (arg, args())
        if (fo ^ (arg->type()->isa<Pi>() != 0))
            res[size++] = arg;

    res.shrink(size);

    return res;
}

bool Lambda::is_cascading() const {
    if (is_fo() || uses().size() != 1)
        return false;

    Use use = *uses().begin();
    return !use.def()->isa<Lambda>() || !use.index() > 0;
}

Array<const Param*> Lambda::fo_params() const { return classify_params<true>(); }
Array<const Param*> Lambda::ho_params() const { return classify_params<false>(); }
Array<const Def*> Lambda::fo_args() const { return classify_args<true>(); }
Array<const Def*> Lambda::ho_args() const { return classify_args<false>(); }
bool Lambda::is_fo()  const { return pi()->is_fo(); }
bool Lambda::is_ho() const { return pi()->is_ho(); }

void Lambda::jump(const Def* to, ArrayRef<const Def*> args) {
    alloc(args.size() + 1);
    set_op(0, to);

    size_t x = 1;
    for_all (arg, args)
        set_op(x++, arg);
}

void Lambda::branch(const Def* cond, const Def* tto, const Def*  fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<const Def*>(0, 0));
}

Lambda* Lambda::drop(size_t i, const Def* with) {
    const Def* awith[] = { with };
    size_t args[] = { i };

    return drop(args, awith);
}

Lambda* Lambda::drop(ArrayRef<const Def*> with) {
    if (with.empty())
        return this;

    Array<size_t> args(with.size());
    for (size_t i = 0, e = args.size(); i < e; ++i)
        args[i] = i;

    return drop(args, with);
}

class Dropper {
public:

    typedef boost::unordered_map<const Def*, const Def*> Old2New;

    Dropper(Lambda* olambda, ArrayRef<size_t> args, ArrayRef<const Def*> with)
        : scope(olambda)
        , args(args)
        , with(with)
        , world(olambda->world())
    {}

    Lambda* drop();
    void drop_head(Lambda* olambda);
    void drop_body(Lambda* olambda, Lambda* nlambda);
    const Def* drop_target(const Def* to);
    void drop(const PrimOp* primop);

    Scope scope;
    ArrayRef<size_t> args;
    ArrayRef<const Def*> with;
    World& world;
    Lambda* nentry;
    Lambda* oentry;
    Old2New old2new;
};

Lambda* Lambda::drop(ArrayRef<size_t> args, ArrayRef<const Def*> with) {
    if (with.empty())
        return this;

    Dropper dropper(this, args, with);
    return dropper.drop();
}

Lambda* Dropper::drop() {
    oentry = scope.entry();
    const Pi* o_pi = oentry->pi();

    size_t o_numargs = o_pi->size();
    size_t numdrop = args.size();
    size_t n_numargs = o_numargs - numdrop;

    Array<const Type*> elems(n_numargs);

    for (size_t i = 0, a = 0, e = 0; i < o_numargs; ++i) {
        if (a < o_numargs && args[a] == i)
            ++a;
        else
            elems[e++] = o_pi->elem(i);
    }

    const Pi* n_pi = world.pi(elems);
    nentry = world.lambda(n_pi);
    nentry->debug = oentry->debug + ".dropped";

    // put in params for entry (oentry)
    for (size_t i = 0, a = 0, n = 0; i < o_numargs; ++i) {
        const Param* oparam = oentry->param(i);
        if (a < o_numargs && args[a] == i) {
            const Def* w = with[a++];
            old2new[oparam] = w; //with[a++];
        } else {
            const Param* nparam = nentry->param(n++);
            nparam->debug = oparam->debug + ".dropped";
            old2new[oparam] = nparam;
        }
    }

    //old2new[oentry] = nentry;

    // create stubs for all other lambdas and put their params into the map
    for_all (olambda, scope.rpo().slice_back(1)) {
        Lambda* nlambda = olambda->stub();
        nlambda->debug += ".dropped";
        old2new[olambda] = nlambda;

        for (size_t i = 0, e = nlambda->params().size(); i != e; ++i) {
            old2new[olambda->param(i)] = nlambda->param(i);
            nlambda->param(i)->debug += ".dropped";
        }
    }

    drop_body(oentry, nentry);

    for_all (cur, scope.rpo().slice_back(1))
        drop_body(cur, (Lambda*) old2new[cur]);

    return nentry;
}

void Dropper::drop_body(Lambda* olambda, Lambda* nlambda) {
    for_all (param, olambda->params()) {
        for_all (use, param->uses())
            if (const PrimOp* primop = use.def()->isa<PrimOp>())
                drop(primop);
    }

    Array<const Def*> args(olambda->args().size());
    for (size_t i = 0; i < args.size(); ++i) {
        const Def* odef = olambda->arg(i);
        Old2New::iterator iter = old2new.find(odef);
        if (iter != old2new.end())
            args[i] = iter->second;
        else
            args[i] = odef;
    }

    if (olambda->to() == oentry) {
        std::cout << "fdjkfjdsl" << std::endl;
    }

    nlambda->jump(drop_target(olambda->to()), args);
}

void Dropper::drop(const PrimOp* oprimop) {
    Array<const Def*> ops(oprimop->size());

    size_t i = 0; 
    for_all (op, oprimop->ops()) {
        Old2New::iterator iter = old2new.find(op);
        if (iter != old2new.end())
            ops[i++] = iter->second;
        else if (op->is_const())
            ops[i++] = op;
        else if (const Param* oparam = op->isa<Param>()) {
            if (scope.contains(oparam->lambda()))
                ops[i++] = op;
        } else if (Lambda* lambda = op->isa_lambda()) {
            LambdaSet::iterator iter = scope.lambdas().find(lambda);
            if (iter == scope.lambdas().end())
                ops[i++] = op;
        }
    }

    PrimOp* nprimop = oprimop->clone();
    nprimop->update(ops);
    old2new[oprimop] = world.consume(nprimop);

    for_all (use, oprimop->uses())
        if (const PrimOp* oprimop = use.def()->isa<PrimOp>())
            drop(oprimop);
}

const Def* Dropper::drop_target(const Def* to) {
    Old2New::iterator iter = old2new.find(to);
    if (iter != old2new.end())
        return iter->second;

    if (const PrimOp* oprimop = to->isa<PrimOp>()) {
        Array<const Def*> ops(oprimop->size());

        size_t i = 0; 
        for_all (op, oprimop->ops()) {
            Old2New::iterator iter = old2new.find(op);
            if (iter != old2new.end())
                ops[i++] = iter->second;
            else if (op->is_const())
                ops[i++] = op;
            else
                ops[i++] = drop_target(op);
        }

        PrimOp* nprimop = oprimop->clone();
        nprimop->update(ops);

        return old2new[oprimop] = world.consume(nprimop);
    }

    return to;
}

} // namespace anydsl
