#include "anydsl2/analyses/scope.h"

#include "anydsl2/lambda.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

static void jump_to_param_users(LambdaSet& scope, Lambda* lambda);
static void walk_up(LambdaSet& scope, Lambda* lambda);
static void find_user(LambdaSet& scope, const Def* def);

LambdaSet find_scope(Lambda* lambda) {
    LambdaSet scope;
    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    return scope;
}

static void jump_to_param_users(LambdaSet& scope, Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(scope, param);
}

static void find_user(LambdaSet& scope, const Def* def) {
    if (Lambda* lambda = def->isa_lambda())
        walk_up(scope, lambda);
    else {
        for_all (use, def->uses())
            find_user(scope, use.def());
    }
}

static void walk_up(LambdaSet& scope, Lambda* lambda) {
    if (scope.find(lambda) != scope.end())
        return;// already inside scope so break

    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    for_all (pred, lambda->preds())
        walk_up(scope, pred);
}

size_t Scope::number(const LambdaSet& lambdas, Lambda* cur, size_t i) {
    // mark as visited
    cur->sid_ = 0;

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (lambdas.find(succ) != lambdas.end() && succ->sid_invalid())
            i = number(lambdas, succ, i);
    }

    return (cur->sid_ = i) + 1;
}

Scope::Scope(Lambda* entry) {
    lambdas_ = find_scope(entry);

    for_all (lambda, lambdas_)
        lambda->invalidate_sid();

    size_t num = number(lambdas_, entry, 0);
    rpo_.alloc(num);
    preds_.alloc(num);
    succs_.alloc(num);

    // remove unreachable lambdas from set and fix numbering
    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        LambdaSet::iterator j = i++;
        Lambda* lambda = *j;
        if (lambda->sid() >= num) { // lambda is unreachable
            lambdas_.erase(j);
            continue; 
        }

        lambda->sid_ = num - 1 - lambda->sid_;
    }

    for_all (lambda, lambdas_) {
        size_t sid = lambda->sid();
        rpo_[sid] = lambda;

        {
            Lambdas& succs = succs_[sid];
            succs.alloc(lambda->succs().size());
            size_t i = 0;
            for_all (succ, lambda->succs()) {
                if (lambdas_.find(succ) != lambdas_.end())
                    succs[i++] = succ;
            }
            succs.shrink(i);
        }
        {
            Lambdas& preds = preds_[sid];
            preds.alloc(lambda->preds().size());
            size_t i = 0;
            for_all (pred, lambda->preds()) {
                if (lambdas_.find(pred) != lambdas_.end())
                    preds[i++] = pred;
            }
            preds.shrink(i);
        }
    }

    assert(rpo_[0] == entry && "bug in numbering");
    assert(rpo_.size() == lambdas_.size());
}

const Scope::Lambdas& Scope::preds(Lambda* lambda) const {
    assert(contains(lambda)); 
    return preds_[lambda->sid()]; 
}

const Scope::Lambdas& Scope::succs(Lambda* lambda) const {
    assert(contains(lambda)); 
    return succs_[lambda->sid()]; 
}

void Scope::reassign_sids() {
    for (size_t i = 0, e = rpo().size(); i != e; ++i)
        rpo_[i]->sid_ = i;
}

//------------------------------------------------------------------------------

class FVFinder {
public:

    FVFinder(const Scope& scope, FreeVariables& fv)
        : scope(scope)
        , fv(fv)
        , world(scope.world())
        , pass(world.new_pass())
    {}

    bool map(const Def* def, bool within) {
        assert(!def->is_visited(pass));
        def->visit(pass);
        def->flags[0] = within;
        def->flags[1] = false;
        return within;
    }
    bool is_within(const Def* def) { assert(def->is_visited(pass)); return def->flags[0]; } 
    bool is_queued(const Def* def) { assert(def->is_visited(pass)); return def->flags[1]; }
    void queue(const Def* def) { assert(def->is_visited(pass)); def->flags[1] = true; fv.push_back(def); }
    void find();
    void find(const Lambda* lambda);
    bool find(const Def* def);

private:

    const Scope& scope;
    FreeVariables& fv;
    World& world;
    size_t pass;
};

FreeVariables Scope::free_variables() const { 
    FreeVariables fv; 
    FVFinder(*this, fv).find();
    return fv;
}

void FVFinder::find() {
    for_all (lambda, scope.rpo())
        find(lambda);
}

void FVFinder::find(const Lambda* lambda) {
    for_all (op, lambda->ops())
        find(op);
}

bool FVFinder::find(const Def* def) {
    if (def->is_visited(pass))
        return is_within(def);

    if (def->is_const())
        return map(def, true);

    if (const Param* param = def->isa<Param>())
        return map(param, scope.contains(param->lambda()));


    bool within = false;
    for_all (op, def->ops())
        within |= find(op);

    if (within) {
        for_all (op, def->ops())
            if (!is_within(op) && !is_queued(op))
                queue(op);
    }

    return map(def, within);
}

//------------------------------------------------------------------------------

class Mapper {
public:

    Mapper(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
           ArrayRef<const Def*> to_lift, bool self, const GenericMap& generic_map)
        : scope(scope)
        , to_drop(to_drop)
        , drop_with(drop_with)
        , to_lift(to_lift)
        , generic_map(generic_map)
        , world(scope.world())
        , pass(world.new_pass())
        , self(self)
    {}

    Lambda* mangle();
    void map_body(Lambda* olambda, Lambda* nlambda);
    Lambda* map_head(Lambda* olambda);
    const Def* drop(const Def* odef);
    const Def* map(const Def* def, const Def* to) {
        assert(!def->is_visited(pass));
        def->visit(pass);
        def->cptr = to;
        return to;
    }
    const Def* lookup(const Def* def) {
        assert(def->is_visited(pass));
        return (const Def*) def->cptr;
    }

    const Scope& scope;
    ArrayRef<size_t> to_drop;
    ArrayRef<const Def*> drop_with;
    ArrayRef<const Def*> to_lift;
    GenericMap generic_map;
    World& world;
    size_t pass;
    Lambda* nentry;
    Lambda* oentry;
    bool self;
};

//------------------------------------------------------------------------------

Lambda* Scope::drop(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, bool self, const GenericMap& generic_map) {
    return mangle(to_drop, drop_with, Array<const Def*>(), self, generic_map);
}

Lambda* Scope::lift(ArrayRef<const Def*> to_lift, bool self, const GenericMap& generic_map) {
    return mangle(Array<size_t>(), Array<const Def*>(), to_lift, self, generic_map);
}

Lambda* Scope::mangle(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                       ArrayRef<const Def*> to_lift, bool self, const GenericMap& generic_map) {
    return Mapper(*this, to_drop, drop_with, to_lift, self, generic_map).mangle();
}

//------------------------------------------------------------------------------

Lambda* Mapper::mangle() {
    oentry = scope.entry();
    const Pi* o_pi = oentry->pi();
    Array<const Type*> nelems = o_pi->elems().cut(to_drop, to_lift.size());
    size_t offset = o_pi->elems().size() - to_drop.size();

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x)
        nelems[i] = to_lift[x]->type();

    const Pi* n_pi = world.pi(nelems)->specialize(generic_map)->as<Pi>();
    nentry = world.lambda(n_pi, oentry->name + ".d");

    // put in params for entry (oentry)
    // op -> iterates over old params
    // np -> iterates over new params
    //  i -> iterates over to_drop
    for (size_t op = 0, np = 0, i = 0, e = o_pi->size(); op != e; ++op) {
        const Param* oparam = oentry->param(op);
        if (i < to_drop.size() && to_drop[i] == op)
            map(oparam, drop_with[i++]);
        else {
            const Param* nparam = nentry->param(np++);
            nparam->name = oparam->name + ".d";
            map(oparam, nparam);
        }
    }

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x) {
        map(to_lift[x], nentry->param(i));
        nentry->param(i)->name = to_lift[x]->name;
    }

    map(oentry, oentry);
    map_body(oentry, nentry);

    for_all (cur, scope.rpo().slice_back(1)) {
        if (!cur->is_visited(pass))
            map_head(cur);
        map_body(cur, lookup(cur)->as_lambda());
    }

    return nentry;
}

Lambda* Mapper::map_head(Lambda* olambda) {
    assert(!olambda->is_visited(pass));
    Lambda* nlambda = olambda->stub(generic_map, olambda->name + ".d");
    map(olambda, nlambda);

    for_all2 (oparam, olambda->params(), nparam, nlambda->params()) {
        map(oparam, nparam);
        nparam->name += ".d";
    }

    return nlambda;
}

void Mapper::map_body(Lambda* olambda, Lambda* nlambda) {
    Array<const Def*> ops(olambda->ops().size());
    for (size_t i = 0, e = ops.size(); i != e; ++i)
        ops[i] = drop(olambda->op(i));

    ArrayRef<const Def*> nargs(ops.slice_back(1));  // new args of nlambda
    const Def* ntarget = ops.front();               // new target of nlambda

    // check whether we can optimize tail recursion
    if (self && ntarget == oentry) {
        bool substitute = true;
        for (size_t i = 0, e = to_drop.size(); i != e && substitute; ++i)
            substitute &= nargs[to_drop[i]] == drop_with[i];

        if (substitute)
            return nlambda->jump(nentry, nargs.cut(to_drop));
    }

    nlambda->jump(ntarget, nargs);
}

const Def* Mapper::drop(const Def* odef) {
    if (odef->is_visited(pass))
        return lookup(odef);

    if (Lambda* olambda = odef->isa_lambda()) {
        if (scope.contains(olambda)) {
            assert(scope.lambdas().size() == scope.rpo().size());
            return map_head(olambda);
        } else
            return map(odef, odef);
    } else if (odef->isa<Param>())
        return map(odef, odef);

    bool is_new = false;
    const PrimOp* oprimop = odef->as<PrimOp>();
    Array<const Def*> nops(oprimop->size());
    for_all2 (&nop, nops, op, oprimop->ops()) {
        nop = drop(op);
        is_new |= nop != op;
    }

    return map(oprimop, is_new ? world.primop(oprimop, nops) : oprimop);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
