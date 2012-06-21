#include "anydsl/cfg.h"

#include "anydsl/lambda.h"
#include "anydsl/type.h"
#include "anydsl/var.h"
#include "anydsl/world.h"

namespace anydsl {

BB::BB(Fct* fct, const std::string& debug /*= ""*/) 
    : sealed_(false)
    , fct_(fct)
    , topLambda_(new Lambda())
    , curLambda_(topLambda_)
{
    topLambda_->debug = debug;
}

Var* BB::setVar(const Symbol& symbol, const Def* def) {
    anydsl_assert(values_.find(symbol) == values_.end(), "double insert");

    Var* lvar = new Var(symbol, def);
    values_[symbol] = lvar;

    return lvar;
}

Var* BB::getVar(const Symbol& symbol) {
    BB::ValueMap::iterator i = values_.find(symbol);

    // if var is known -> return current var
    if (i != values_.end())
        return i->second;

    if (!sealed_ || preds_.size() > 1) {
        // otherwise insert a 'phi', i.e., create a param and remember to fix the callers
        Param* param = topLambda_->appendParam();
        size_t index = in_.size();
        in_.push_back(param);
        Var* lvar = setVar(symbol, param);
        todos_[symbol] = index;

        return lvar;
    }
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    Var* lvar = pred->getVar(symbol);

    // create copy of lvar in this BB
    return setVar(lvar->symbol, lvar->def);
}

void BB::seal() {
    anydsl_assert(!sealed(), "already finalized");

    sealed_ = true;

    for_all (p, todos_) {
        for_all (pred, preds_) {
            const Symbol& symbol = p.first;
            size_t x = p.second;
            Defs& out = pred->out_;

            // make potentially room for the new arg
            out.resize(std::max(x + 1, out.size()));

            anydsl_assert(!pred->out_[x], "already set");
            pred->out_[x] = pred->getVar(symbol)->def;
        }
    }
}

void BB::goesto(BB* to) {
    assert(to);

    to_ = to->topLambda();
    this->flowsto(to);

    anydsl_assert(this->succs().size() == 1, "wrong number of succs");
}

void BB::branches(const Def* cond, BB* tbb, BB* fbb) {
    assert(tbb);
    assert(fbb);

    to_ = world().createSelect(cond, tbb->topLambda(), fbb->topLambda());
    this->flowsto(tbb);
    this->flowsto(fbb);

    anydsl_assert(succs().size() == 2, "wrong number of succs");
}

void BB::fixto(BB* to) {
    if (!to_)
        this->goesto(to);
}

void BB::flowsto(BB* to) {
    BBs::iterator i = succs_.find(to);
    anydsl_assert(!to->sealed(), "'to' already sealed");


    if (i == succs_.end()) {
        succs_.insert(to);
        to->preds_.insert(this);
    } else {
        anydsl_assert(to->preds_.find(this) != to->preds_.end(), "flow out of sync");
        /* do nothing */
    }
}

World& BB::world() {
    return topLambda_->world();
}

//------------------------------------------------------------------------------

Fct::Fct(const Symbol& symbol, const Pi* pi)
    : pi_(pi)
    , lambda_(pi->world().createLambda(pi))
{}

BB* Fct::createBB(const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, debug);
    cfg_.insert(bb);

    return bb;
}

} // namespace anydsl
