#include "anydsl2/analyses/scope.h"

#include <algorithm>

#include "anydsl2/lambda.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/loopforest.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

class ScopeBuilder {
public:

    ScopeBuilder(Lambda* entry, Scope* scope);

    bool is_visited(Lambda* lambda) { return lambda->is_visited(pass); }
    bool contains(Lambda* lambda) { return lambda->scope() == scope; }
    void insert(Lambda* lambda) { lambda->visit_first(pass); lambda->scope_ = scope; scope->rpo_.push_back(lambda); }

    void jump_to_param_users(Lambda* lambda);
    void up(Lambda* lambda);
    void find_user(const Def* def);
    size_t number(Lambda* cur, size_t i);

private:

    Scope* scope;
    size_t pass;
};

struct ScopeLess {
    bool operator () (const Lambda* l1, const Lambda* l2) const { return l1->sid() < l2->sid(); }
};

ScopeBuilder::ScopeBuilder(Lambda* entry, Scope* scope)
    : scope(scope)
    , pass(entry->world().new_pass())
{
    scope->rpo_.reserve(scope->size());
    insert(entry);
    jump_to_param_users(entry);
    pass = entry->world().new_pass();
    size_t num = number(entry, 0);
    assert(num <= scope->rpo_.size());

    for_all (lambda, scope->rpo_) {
        if (lambda->is_visited(pass)) {
            lambda->sid_ = num - 1 - lambda->sid_;
        } else {
            lambda->scope_ = 0;
            lambda->sid_ = size_t(-1);
        }
    }
    
    std::sort(scope->rpo_.begin(), scope->rpo_.end(), ScopeLess());
    scope->rpo_.resize(num);
}

void ScopeBuilder::jump_to_param_users(Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(param);
}

void ScopeBuilder::find_user(const Def* def) {
    if (Lambda* lambda = def->isa_lambda())
        up(lambda);
    else {
        for_all (use, def->uses())
            find_user(use.def());
    }
}

void ScopeBuilder::up(Lambda* lambda) {
    if (is_visited(lambda))
        return;

    insert(lambda);
    jump_to_param_users(lambda);

    for_all (pred, lambda->preds())
        up(pred);
}

size_t ScopeBuilder::number(Lambda* cur, size_t i) {
    cur->visit_first(pass);

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (contains(succ) && !succ->is_visited(pass))
            i = number(succ, i);
    }

    return (cur->sid_ = i) + 1;
}

Scope::~Scope() {
    for_all (lambda, rpo_)
        lambda->scope_ = 0;
}

Scope::Scope(Lambda* entry) {
    ScopeBuilder(entry, this);
    size_t num = rpo_.size();
    preds_.alloc(num);
    succs_.alloc(num);

    for_all (lambda, rpo_) {
        size_t sid = lambda->sid();
        rpo_[sid] = lambda;

        {
            Array<Lambda*>& succs = succs_[sid];
            succs.alloc(lambda->succs().size());
            size_t i = 0;
            for_all (succ, lambda->succs()) {
                if (contains(succ))
                    succs[i++] = succ;
            }
            succs.shrink(i);
        }
        {
            Array<Lambda*>& preds = preds_[sid];
            preds.alloc(lambda->preds().size());
            size_t i = 0;
            for_all (pred, lambda->preds()) {
                if (contains(pred))
                    preds[i++] = pred;
            }
            preds.shrink(i);
        }
    }

    assert(rpo_[0] == entry && "bug in numbering");
}

ArrayRef<Lambda*> Scope::preds(Lambda* lambda) const {
    assert(contains(lambda)); 
    return preds_[lambda->sid()]; 
}

ArrayRef<Lambda*> Scope::succs(Lambda* lambda) const {
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
        def->visit_first(pass);
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
        def->visit_first(pass);
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
        if (scope.contains(olambda))
            return map_head(olambda);
        else
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

Lambda* Scope::clone(bool self, const GenericMap& generic_map) { 
    return mangle(Array<size_t>(), Array<const Def*>(), Array<const Def*>(), self, generic_map);
}

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

const DomTree& Scope::domtree() const { 
    return domtree_ ? *domtree_ : *(domtree_ = new DomTree(*this));
}

const LoopForestNode* Scope::loopforest() const { 
    return loopforest_ ? loopforest_ : loopforest_ = create_loop_forest(*this);
}

const LoopInfo& Scope::loopinfo() const { 
    return loopinfo_ ? *loopinfo_ : *(loopinfo_ = new LoopInfo(*this));
}

//------------------------------------------------------------------------------

} // namespace anydsl2
