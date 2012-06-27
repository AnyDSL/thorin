#include "anydsl/cfg.h"

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
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

Var* BB::getVar(const Symbol& symbol, const Type* type) {
    BB::VarMap::iterator i = values_.find(symbol);

    // if var is known -> return current var
    if (i != values_.end())
        return i->second;

    // value is undefined
    if (fct_ == this) {
        // TODO provide hook instead of fixed functionality
        std::cerr << "'" << symbol << "'" << " may be undefined" << std::endl;
        return setVar(symbol, world().undef(type));
    }

    // insert a 'phi', i.e., create a param and remember to fix the callers
    if (!sealed_ || preds_.size() > 1) {
        const Param* param = topLambda_->appendParam(type);
        size_t index = in_.size();
        in_.push_back(param);
        Var* lvar = setVar(symbol, param);
        todos_[symbol] = Todo(index, type);

        return lvar;
    }
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    Var* lvar = pred->getVar(symbol, type);

    // create copy of lvar in this BB
    return setVar(lvar->symbol, lvar->def);
}

void BB::seal() {
    anydsl_assert(!sealed(), "already finalized");

    sealed_ = true;

    for_all (p, todos_) {
        for_all (pred, preds_) {
            const Symbol& symbol = p.first;
            size_t index = p.second.index;
            const Type* type = p.second.type;

            Defs& out = pred->out_;

            // make potentially room for the new arg
            if (index >= out.size())
                out.resize(index + 1);

            anydsl_assert(!pred->out_[index], "already set");
            pred->out_[index] = pred->getVar(symbol, type)->def;
        }
    }
}

void BB::goesto(BB* to) {
    assert(to);
    this->flowsto(to);
    anydsl_assert(this->succs().size() == 1, "wrong number of succs");
}

void BB::branches(const Def* cond, BB* tbb, BB* fbb) {
    assert(tbb);
    assert(fbb);

    cond_ = cond;
    tlambda_ = tbb->topLambda_;
    flambda_ = fbb->topLambda_;
    this->flowsto(tbb);
    this->flowsto(fbb);

    anydsl_assert(succs().size() == 2, "wrong number of succs");
}

void BB::fixto(BB* to) {
    if (succs().empty())
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
    return fct_->world();
}

void BB::emit() {
    const Jump* jump;
    switch (succs().size()) {
        case 1:
            jump = world().createJump((*succs().begin())->topLambda(), out_.begin().base(), out_.end().base());
            break;
        case 2:
            jump = world().createBranch(cond_, tlambda_, flambda_, out_.begin().base(), out_.end().base());
            break;
        default: 
            ANYDSL_UNREACHABLE;
    }

    topLambda_->setJump(jump);
    world().finalize(topLambda_);
}

void BB::calcType() {
    topLambda_->calcType(world());
}

//------------------------------------------------------------------------------

Fct::Fct(World& world, const FctParams& fparams, const Type* retType, const std::string& debug /*= ""*/) 
    : retType_(retType)
    , world_(world)
{
    sealed_ = true;
    fct_ = this;
    curLambda_ = topLambda_ = new Lambda();
    topLambda_->debug = debug;

    for_all (p, fparams)
        setVar(p.symbol, topLambda_->appendParam(p.type));

    if (retType)
        setVar(Symbol("<return>"), world.pi1(retType));
}


BB* Fct::createBB(const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, debug);
    cfg_.insert(bb);

    return bb;
}

void Fct::emit() {
    topLambda_->calcType(world());

    for_all (bb, cfg_)
        bb->calcType();

    for_all (bb, cfg_)
        bb->emit();
}

} // namespace anydsl
