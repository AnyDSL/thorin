#include "anydsl/cfg.h"

#include "anydsl/lambda.h"
#include "anydsl/type.h"
#include "anydsl/var.h"
#include "anydsl/world.h"

namespace anydsl {

BB::BB(Fct* fct, bool final, const std::string& debug /*= ""*/) 
    : final_(final)
    , fct_(fct)
    , topLambda_(new Lambda())
    , curLambda_(topLambda_)
{
    topLambda_->debug = debug;
}

void BB::setVar(const LVar& lvar) {
    anydsl_assert(values_.find(lvar.symbol()) == values_.end(), "double insert");
    values_[lvar.symbol()] = lvar;
}

LVar BB::getVar(const Symbol& symbol) {
    BB::ValueMap::iterator i = values_.find(symbol);

    // if var is known -> return current var
    if (i != values_.end())
        return i->second;

    if (!final_ || preds_.size() > 1) {
        // otherwise insert a 'phi', i.e., create a param and remember to fix the callers
        Param* param = topLambda_->appendParam();
        size_t index = in_.size();
        in_.push_back(param);
        LVar lvar(param, symbol);
        setVar(lvar);
        todos_[symbol] = index;

        return lvar;
    }
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    LVar lvar = pred->getVar(symbol);
    setVar(lvar);

    return lvar;
}

void BB::finalize() {
    anydsl_assert(!final_, "already finalized");

    final_ = true;

    for_all (pred, preds_) {
        for_all (p, todos_) {
            const Symbol& symbol = p.first;
            size_t x = p.second;
            Defs& out = pred->out_;

            // make potentially room for the new arg
            out.resize(std::max(x + 1, out.size()));

            anydsl_assert(!pred->out_[x], "already set");
            pred->out_[x] = getVar(symbol).load();
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

BB* Fct::createBB(bool final, const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, final, debug);
    cfg_.insert(bb);

    return bb;
}

} // namespace anydsl
