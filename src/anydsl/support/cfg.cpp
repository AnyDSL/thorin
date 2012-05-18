#include "anydsl/support/cfg.h"

#include "anydsl/air/literal.h"
#include "anydsl/air/terminator.h"
#include "anydsl/air/world.h"
#include "anydsl/support/binding.h"
#include "anydsl/util/foreach.h"

using namespace anydsl;

namespace anydsl {

//------------------------------------------------------------------------------

BB::BB(World& world, const std::string& name /*= ""*/) 
    : parent_(0)
    , lambda_(world.createLambda(0))
{
    lambda_->debug = name;
}

World& BB::world() { 
    return lambda_->world();
}

void BB::insert(BB* bb) {
    anydsl_assert(!bb->parent_, "parent already set");

    bb->parent_ = this;
    this->lambda_->insert(bb->lambda());
    children_.insert(bb);
}

void BB::goesto(BB* to) {
    assert(to);
    world().createGoto(lambda_, to->lambda_);
    this->flowsTo(to);
    anydsl_assert(succ_.size() == 1, "wrong number of succ");
}

void BB::branches(Def* cond, BB* tbb, BB* fbb) {
    assert(tbb);
    assert(fbb);
    world().createBranch(lambda_, cond, tbb->lambda_, fbb->lambda_);
    this->flowsTo(tbb);
    this->flowsTo(fbb);
    anydsl_assert(succ_.size() == 2, "wrong number of succ");
}

void BB::invokes(Def* fct) {
    anydsl_assert(fct, "must be valid");
    world().createInvoke(lambda(), fct);
    anydsl_assert(succ_.size() == 0, "wrong number of succ");
    // succs by invokes are not captured in the CFG
}

void BB::flowsTo(BB* to) {
    BBs::iterator i = succ_.find(to);
    if (i == succ_.end()) {
        succ_.insert(to);
        to->pred_.insert(this);
    } else {
        anydsl_assert(to->pred_.find(this) != to->pred_.end(), "flow out of sync");
        /* do nothing */
    }
}

#if 0
void BB::finalizeAll() {
    processTodos();

    FOREACH(bb, children_)
        bb->finalizeAll();
}

void BB::processTodos() {
    if (finalized_)
        return;
    finalized_ = true;

#ifdef DEBUG_CFG
    std::cout << name() << std::endl;
#endif
    anydsl_assert(!pred_.empty() || dcast<Fct>(this), "must not be empty");

    FOREACH(i, todos_) {
        size_t x = i.second;
        Symbol sym = i.first;

        FOREACH(pred, pred_) {
            anydsl_assert(!pred->succ_.empty(), "must have at least one succ");
            anydsl_assert(pred->succ_.find(this) != pred->succ_.end(), "incorrectly wired");
            pred->finalize(x, sym);
        }
    }
}

void BB::finalize(size_t x, const Symbol sym) {
    if (Beta* beta = getBeta()) {
        //anydsl_assert(beta->args().empty(), "must be empty");
        fixBeta(beta, x, sym, 0);
    } else if (Branch* branch = getBranch()) {
        Lambda* lam[2] = {scast<Lambda>(branch-> trueExpr.def()), 
                          scast<Lambda>(branch->falseExpr.def()) };

        for (size_t i = 0; i < 2; ++i) {
            Beta* beta = scast<Beta>(lam[i]->body());
            fixBeta(beta, x, sym, 0);
        }
    } else
        ANYDSL_UNREACHABLE;
}

void BB::fixBeta(Beta* beta, size_t x, const Symbol sym, Type* type) {
    UseList& args = beta->args();

    // make room for new arg
    if (x >= args.size())
        args.resize(x+1);

#ifdef DEBUG_CFG
    std::cout << "fixing: " << name() << " pos: " << x << " size: " << args.size() << std::endl;
#endif

    Def* def = getVN(beta->loc(), sym, 0, true)->def;

    anydsl_assert(!args[x].isSet(), "must be unset");
    anydsl_assert(hasVN(sym) || pred_.size() == 1, "must be found");
    anydsl_assert(def, "must be valid");
    args[x] = def;
}

#endif

Binding* BB::getVN(const Symbol sym, const Type* type, bool finalize) {
    BB::ValueMap::iterator i = values_.find(sym);

    if (i == values_.end()) {
        if (pred_.size() == 1) {
            BB* pred = *pred_.begin();
            Binding* bind = pred->getVN(sym, type, finalize);
            // create copy of binding in this block
            Binding* newBind = new Binding(bind->sym, bind->def);
            setVN(newBind);

            anydsl_assert(newBind->def, "must be valid");
            return newBind;
        } else {
            // add bind as param to current BB
            ParamIter param = lambda_->appendParam(type);
            // insert new VN
            Binding* bind = new Binding(sym, *param);
            setVN(bind);

            if (finalize) {
                FOREACH(pred, pred_)
                    pred->finalize(param, sym);
            } else {
                // remember to fix preds
#ifdef DEBUG_CFG
                std::cout << "todo: " << name() << ": " << sym << " -> " << x << std::endl;
                FOREACH(pred, pred_)
                    std::cout << "    pred: " << pred->name() << std::endl;
#endif
                anydsl_assert(todos_.find(sym) == todos_.end(), "double insert");
                todos_[sym] = param;
            }

            anydsl_assert(bind->def, "must be valid");
            return bind;
        }
    }

    anydsl_assert(i->second->def, "must be valid");
    return i->second;
}

void BB::setVN(Binding* bind) {
    anydsl_assert(values_.find(bind->sym) == values_.end(), "double insert");
    values_[bind->sym] = bind;
}

std::string BB::name() const { 
    const std::string& str = lambda_->debug;
    return str.empty() ? "<unnamed>" : str;
}

#if 0
void BB::inheritValues(BB* bb) {
    FOREACH(p, bb->values_) {
        Binding* bind = p.second;
        anydsl_assert(p.first == bind->sym, "symbols must be equal");
        values_[p.first] = new Binding(bind->sym, bind->def);
    }
}
#endif


#ifndef NDEBUG

bool BB::verify(BB* bb) {
    if (this == bb)
        return true;

    FOREACH(i, children_)
        if (i->verify(bb))
            return true;
        
    return false;
}

#endif

//------------------------------------------------------------------------------

#if 0

Fct::Fct(const Location& loc, const Symbol sym)
    : BB(0, loc, sym)
    , ret_(0)
{}

Fct::Fct(BB* parent, const Location& loc, const Symbol sym)
    : BB(parent, loc, sym)
    , ret_(0)
{}

#if 0
Fct* Fct::createSubFct(const Location& loc, const Symbol sym) {
    Fct* subFct = new Fct(this, loc, sym);
    children_.insert(subFct);
    fix_->append(subFct->param_, subFct->lambda_);

    return subFct;
}
#endif

// TODO handle this with normal magic by introducing a <ret> value
void Fct::setReturn(const Location& loc, Type* retType) {
    anydsl_assert(!ret_, "already set");

    ret_ = new Param(loc, Symbol("<return>"));
    ret_->meta.set(Pi::createUnary(retType));

    exit_ = BB::create(Symbol("<exit>"));
    exit_->calls(loc, ret_);

    lambda_->params().push_back(ret_);
}

void Fct::insertReturn(const Location& loc, BB* bb, Def* def) {
    anydsl_assert(bb, "must be valid");
    bb->jumps(loc, exit_);
    bb->getBeta()->args().push_back(def);
}

void Fct::insertCont(const Location& loc, BB* where, Def* cont) {
    anydsl_assert(cont, "must be valid");
    where->calls(loc, cont);
}

Binding* Fct::getVN(const Location& loc, const Symbol sym, Type* type, bool finalize) {
    BB::ValueMap::iterator i = values_.find(sym);
    if (i == values_.end()) {
        Undef* undef = new Undef(loc);
        undef->meta.set(type);
        std::cerr << "may be undefined: " << sym << std::endl;

        return new Binding(sym, undef);
    }

    anydsl_assert(i->second->def, "must be valid");
    return i->second;
}
#endif

//------------------------------------------------------------------------------

/*
 * belongs to
 */

#ifndef NDEBUG

Symbol BB::belongsTo() {
    if (parent_)
        return parent_->belongsTo();
    else 
        return name();
}

#endif

} // namespace impala
