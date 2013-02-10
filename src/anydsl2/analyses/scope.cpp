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

struct ScopeLess {
    bool operator () (const Lambda* l1, const Lambda* l2) const { return l1->sid() < l2->sid(); }
};

Scope::Scope(Lambda* entry) {
    // identify all lambdas depending on entry
    size_t pass = entry->world().new_pass();
    insert(pass, entry);
    jump_to_param_users(pass, entry);

    // number all lambdas in postorder
    pass = entry->world().new_pass();
    size_t num = number(pass, entry, 0);
    assert(num <= rpo().size());
    assert(num >= 1);

    // convert postorder number to reverse postorder number
    for_all (lambda, rpo()) {
        if (lambda->is_visited(pass)) {
            lambda->sid_ = num - 1 - lambda->sid_;
        } else { // lambda is unreachable
            lambda->scope_ = 0;
            lambda->sid_ = size_t(-1);
        }
    }
    
    // sort rpo according to rpo
    std::sort(rpo_.begin(), rpo_.end(), ScopeLess());
    assert(rpo_[0] == entry && "bug in numbering");

    // discard unreachable lambdas;
    rpo_.resize(num);

    // cache preds and succs
    preds_.alloc(num);
    succs_.alloc(num);
    for_all (lambda, rpo_) {
        size_t sid = lambda->sid();
        fill_succ_pred(lambda->succs(), succs_[sid]);
        fill_succ_pred(lambda->preds(), preds_[sid]);
    }
}

template<class T>
inline void Scope::fill_succ_pred(const Lambdas& l_succs_preds, T& succs_preds) {
    succs_preds.alloc(l_succs_preds.size());
    size_t i = 0;
    for_all (item, l_succs_preds) {
        if (contains(item))
            succs_preds[i++] = item;
    }
    succs_preds.shrink(i);
}

Scope::~Scope() {
    for_all (lambda, rpo_)
        lambda->scope_ = 0;
}

void Scope::jump_to_param_users(const size_t pass, Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(pass, param);
}

void Scope::find_user(const size_t pass, const Def* def) {
    if (Lambda* lambda = def->isa_lambda())
        up(pass, lambda);
    else {
        for_all (use, def->uses())
            find_user(pass, use.def());
    }
}

void Scope::up(const size_t pass, Lambda* lambda) {
    if (lambda->is_visited(pass))
        return;

    insert(pass, lambda);
    jump_to_param_users(pass, lambda);

    for_all (pred, lambda->preds())
        up(pass, pred);
}

size_t Scope::number(const size_t pass, Lambda* cur, size_t i) {
    cur->visit_first(pass);

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (contains(succ) && !succ->is_visited(pass))
            i = number(pass, succ, i);
    }

    return (cur->sid_ = i) + 1;
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
    const size_t pass;
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
           ArrayRef<const Def*> to_lift, const GenericMap& generic_map)
        : scope(scope)
        , to_drop(to_drop)
        , drop_with(drop_with)
        , to_lift(to_lift)
        , generic_map(generic_map)
        , world(scope.world())
        , pass(world.new_pass())
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
    const size_t pass;
    Lambda* nentry;
    Lambda* oentry;
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
    nentry = world.lambda(n_pi, oentry->name);

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
            nparam->name = oparam->name;
            map(oparam, nparam);
        }
    }

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x) {
        map(to_lift[x], nentry->param(i));
        nentry->param(i)->name = to_lift[x]->name;
    }

    map(oentry, oentry);
    map_body(oentry, nentry);

    // TODO omit unreachable lambdas
    for_all (cur, scope.rpo().slice_back(1)) {
        if (!cur->is_visited(pass))
            map_head(cur);
        map_body(cur, lookup(cur)->as_lambda());
    }

    return nentry;
}

Lambda* Mapper::map_head(Lambda* olambda) {
    assert(!olambda->is_visited(pass));
    Lambda* nlambda = olambda->stub(generic_map, olambda->name);
    map(olambda, nlambda);

    for_all2 (oparam, olambda->params(), nparam, nlambda->params())
        map(oparam, nparam);

    return nlambda;
}

void Mapper::map_body(Lambda* olambda, Lambda* nlambda) {
    Array<const Def*> ops(olambda->ops().size());
    for (size_t i = 0, e = ops.size(); i != e; ++i)
        ops[i] = drop(olambda->op(i));

    ArrayRef<const Def*> nargs(ops.slice_back(1));  // new args of nlambda
    const Def* ntarget = ops.front();               // new target of nlambda

    // check whether we can optimize tail recursion
    if (ntarget == oentry) {
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

    return map(oprimop, is_new ? world.rebuild(oprimop, nops) : oprimop);
}

//------------------------------------------------------------------------------

Lambda* Scope::clone(const GenericMap& generic_map) { 
    return mangle(Array<size_t>(), Array<const Def*>(), Array<const Def*>(), generic_map);
}

Lambda* Scope::drop(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, const GenericMap& generic_map) {
    return mangle(to_drop, drop_with, Array<const Def*>(), generic_map);
}

Lambda* Scope::lift(ArrayRef<const Def*> to_lift, const GenericMap& generic_map) {
    return mangle(Array<size_t>(), Array<const Def*>(), to_lift, generic_map);
}

Lambda* Scope::mangle(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                       ArrayRef<const Def*> to_lift, const GenericMap& generic_map) {
    return Mapper(*this, to_drop, drop_with, to_lift, generic_map).mangle();
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
