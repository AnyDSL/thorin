#include "impala/cfg.h"

#include "impala/binding.h"
#if 0

using namespace anydsl;

namespace impala {

//------------------------------------------------------------------------------

BB::BB(BB* parent, const Location& loc, const Symbol sym /*= Symbol("")*/) 
    : parent_(parent)
    , param_(new Param(loc, sym))
    , lambda_(new Lambda(loc))
    , fix_(new Fix(loc))
    , finalized_(false)
    , hasTerminator_(false)
{
    fix_->appendNote(new CFGFixAnnotation());

    // type for this BB
    lambda_->meta = Pi::create(loc);
    param_ ->meta = new Tau(loc);

    // embed Fix in lambda
    lambda_->body() = fix_;

    lcursor_ = new Lambda(loc);
    lcursor_->meta = Pi::create(loc);
    fix_->body.set(lcursor_);
}

/*static*/ BB* BB::create(const Symbol sym /*= Symbol("")*/) {
    BB* bb = new BB(0, Location(), sym);
    return bb;
}

Def* BB::appendLambda(CExpr* cexpr, Type* type) {
    anydsl_assert(parent_, "not set");
    anydsl_assert(!lcursor_->body(), "must be empty");

    const Location& loc = cexpr->loc();
    Lambda* lambda = new Lambda(loc);
    Param* result = lambda->appendParam(type, loc);
    cexpr->appendLambda(lambda);
    lcursor_->body() = cexpr;
    lcursor_ = lambda;

    return result;
}

BB* BB::createSubBB(const Location& loc, const Symbol sym /*= Symbol("")*/) {
    BB* subBB = new BB(this, loc, sym);
    children_.insert(subBB);
    fix_->append(subBB->param_, subBB->lambda_);

    return subBB;
}

void BB::hangInBB(BB* bb) {
    anydsl_assert(!bb->parent_, "parent already set");
    bb->parent_ = this;
    children_.insert(bb);
    fix_->append(bb->param_, bb->lambda_);
}

void BB::calls(const Location& loc, Def* f) {
    setTerminator();
    anydsl_assert(!lcursor_->body(), "must be empty");

    Beta* beta = new Beta(loc);
    beta->fct.set(f);
    lcursor_->body() = beta;
    // we can now overwrite lcursor_ in the union
    beta_ = beta;
}

void BB::jumps(const Location& loc, BB* to) {
    setTerminator();
    anydsl_assert(to, "must be valid");
    anydsl_assert(!lcursor_->body(), "must be empty");

    Beta* beta = new Beta(loc);
    beta->fct.set(to->param_);
    lcursor_->body() = beta;
    // we can now overwrite lcursor_ in the union
    beta_ = beta;

    this->flowsTo(to);
    anydsl_assert(succ_.size() == 1, "wrong number of succ");
}

void BB::branches(const Location& loc, Def* cond, BB* toT, BB* toF) {
    setTerminator();
    Branch* branch = new Branch(loc);
    branch->value.set(cond);

    for (size_t i = 0; i < 2; ++i) {
        Lambda* lam = new Lambda(loc);
        Beta* beta = new Beta(loc);
        beta->fct.set(i == 0 ? toT->param_ : toF->param_);
        lam->meta = Pi::create(loc);
        lam->body() = beta;
        branch->appendLambda(lam);
    }


    lcursor_->body() = branch;
    // we can now overwrite lcursor_ in the union
    branch_ = branch; 

    this->flowsTo(toT);
    this->flowsTo(toF);
    anydsl_assert(succ_.size() == 2, "wrong number of succ");
}

void BB::mergeChild() {
    BB* child = *children_.begin();

    // check dom propertiers
    anydsl_assert(children_.size() == 1, "must have exactly one child");
    anydsl_assert(child->parent_ == this, "wrong parent");

    // check cfg properties
    anydsl_assert(succ_.size() == 1, "must have exactly one successor");
    anydsl_assert(*succ_.begin() == child, "wrong successor");
    anydsl_assert(child->pred_.size() == 1, "must have exactly one predessor");
    anydsl_assert(*child->pred_.begin() == this, "wrong predessor");

    //// repair cfg
    //succ_.clear();
    //std::copy(child->succ_.begin(), child->succ_.end(), 
            //std::inserter(this->succ_, this->succ_.begin()));

    anydsl_assert(false, "TODO");
    //delete
    //this->flowsTo(child);
}

void BB::flowsTo(BB* to) {
    BBList::iterator i = succ_.find(to);
    if (i == succ_.end()) {
        succ_.insert(to);
        to->pred_.insert(this);
    } else {
        anydsl_assert(to->pred_.find(this) != to->pred_.end(), "flow out of sync");
        /* do nothing */
    }
}

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

Binding* BB::getVN(const Location& loc, const Symbol sym, anydsl::Type* type, bool finalize) {
    BB::ValueMap::iterator i = values_.find(sym);

    if (i == values_.end()) {
        if (pred_.size() == 1) {
            BB* pred = *pred_.begin();
            Binding* bind = pred->getVN(loc, sym, type, finalize);
            // create copy of binding in this block
            Binding* newBind = new Binding(bind->sym, bind->def);
            setVN(loc, newBind);

            anydsl_assert(newBind->def, "must be valid");
            return newBind;
        } else {
            // add bind as param to current BB
            Param* param = lambda_->appendParam(type, loc);
            size_t x = lambda_->params().size() - 1;
            param->meta = type;
            // insert new VN
            Binding* bind = new Binding(sym, param);
            setVN(loc, bind);

            if (finalize) {
                FOREACH(pred, pred_)
                    pred->finalize(x, sym);
            } else {
                // remember to fix preds
#ifdef DEBUG_CFG
                std::cout << "todo: " << name() << ": " << sym << " -> " << x << std::endl;
                FOREACH(pred, pred_)
                    std::cout << "    pred: " << pred->name() << std::endl;
#endif
                anydsl_assert(todos_.find(sym) == todos_.end(), "double insert");
                todos_[sym] = x;
            }

            anydsl_assert(bind->def, "must be valid");
            return bind;
        }
    }

    anydsl_assert(i->second->def, "must be valid");
    return i->second;
}

void BB::setVN(const anydsl::Location& loc, Binding* bind) {
    anydsl_assert(values_.find(bind->sym) == values_.end(), "double insert");
    values_[bind->sym] = bind;
}

const char* BB::name() { 
    Symbol sym = param_->symbol().str(); 
    if (sym.isEmpty())
        return "<unnamed>";
    else
        return sym.str();
}

Beta* BB::getBeta() {
    if (succ_.size() == 1) 
        return beta_;

    return 0;
}

Branch* BB::getBranch() {
    if (succ_.size() == 2)
        return branch_;

    return 0;
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

Fct::Fct(const Location& loc, const Symbol sym)
    : BB(0, loc, sym)
    , ret_(0)
{}

Fct::Fct(BB* parent, const Location& loc, const Symbol sym)
    : BB(parent, loc, sym)
    , ret_(0)
{}

Fct* Fct::createSubFct(const Location& loc, const Symbol sym) {
    Fct* subFct = new Fct(this, loc, sym);
    children_.insert(subFct);
    fix_->append(subFct->param_, subFct->lambda_);

    return subFct;
}

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

Def* Fct::appendLambda(BB* bb, CExpr* cexpr, Type* type) {
    anydsl_assert(verify(bb), "BB '" 
            + std::string(bb->name()) 
            + "' does not belong to function '"
            + std::string(name()) 
            + "' but to: '" 
            + std::string(bb->belongsTo().str()) + "'");

    return bb->appendLambda(cexpr, type);
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

#endif
