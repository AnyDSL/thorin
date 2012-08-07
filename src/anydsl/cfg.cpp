#include "anydsl/cfg.h"

#include <algorithm>
#include <iterator>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

namespace anydsl {

BB::BB(Fct* fct, const std::string& debug /*= ""*/) 
    : sealed_(false)
    , fct_(fct)
    , topLambda_(new Lambda(world().pi0()))
    , curLambda_(topLambda_)
{
    topLambda_->debug = debug;
}

BB::~BB() {
    for_all (p, vars_)
        delete p.second;
}

Var* BB::setVar(const Symbol& symbol, const Def* def) {
    anydsl_assert(vars_.find(symbol) == vars_.end(), "double insert");

    Var* lvar = new Var(symbol, def);
    vars_[symbol] = lvar;

    return lvar;
}

Var* BB::getVar(const Symbol& symbol, const Type* type) {
    BB::VarMap::iterator i = vars_.find(symbol);

    // if var is known -> return current var
    if (i != vars_.end())
        return i->second;

    // value is undefined
    if (fct_ == this) {
        // TODO provide hook instead of fixed functionality
        std::cerr << "'" << symbol << "'" << " may be undefined" << std::endl;
        return setVar(symbol, world().bottom(type));
    }

    // insert a 'phi', i.e., create a param and remember to fix the callers
    if (!sealed_ || preds_.size() > 1) {
        const Param* param = topLambda_->appendParam(type);
        size_t index = in_.size();
        in_.push_back(param);
        Var* lvar = setVar(symbol, param);
        param->debug = symbol.str();

        Todo todo(index, type);

        if (sealed_)
            fixTodo(symbol, todo);
        else
            todos_[symbol] = todo;

        return lvar;
    }

    // unreachable code
    if (preds().empty())
        return setVar(symbol, world().bottom(type));
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    Var* lvar = pred->getVar(symbol, type);

    // create copy of lvar in this BB
    return setVar(symbol, lvar->def);
}

void BB::seal() {
    anydsl_assert(!sealed(), "already sealed");

    sealed_ = true;

    // eliminate critical edge
    if (preds().size() >= 2) {
        for_all (pred, preds_) {
            if (pred->succs().size() > 1) {
                // critical edge -> eliminate
                BB* empty = fct_->createBB();
                pred->eraseEdge(this);
                pred->flowsto(empty);
                empty->flowsto(this);
                empty->seal();
                pred = empty;
            }
        }
    }

    for_all (p, todos_)
        fixTodo(p.first, p.second);
}

void BB::fixTodo(const Symbol& symbol, Todo todo) {
    anydsl_assert(sealed(), "must be sealed");

    size_t index = todo.index;
    const Type* type = todo.type;
    const Param* param = in_[index];
    const Def* same = 0;

    // find Horspool-like phis
    for_all (pred, preds_) {
        const Def* def = pred->getVar(symbol, type)->def;

        if (def->isa<Undef>() || def == param || same == def)
            continue;

        if (same)
            goto ok;

        same = def;
    }

    std::cout << "superfluous: " << param->debug << " in block " << param->lambda()->debug << std::endl;
    // superfluous phi params are not yet eliminated -- for this reason build them nevertheless

ok:
    for_all (pred, preds_) {
        anydsl_assert(pred->succs().size() == 1, "critical edge elimination did not work");
        Out& out = pred->out_;

        // make potentially room for the new arg
        if (index >= out.size())
            out.resize(index + 1);

        anydsl_assert(!pred->out_[index], "already set");
        out[index] = pred->getVar(symbol, type)->def;
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
    tbb_ = tbb;
    fbb_ = fbb;
    this->flowsto(tbb);
    this->flowsto(fbb);

    anydsl_assert(succs().size() == 2, "wrong number of succs");
}

const Def* BB::calls(const Def* to, ArrayRef<const Def*> args, const Type* retType) {
    static int id = 0;

    // create next continuation in cascade
    Lambda* next = new Lambda(world().pi0());
    next->debug = curLambda_->debug + "_" + to->debug;
    const Param* result = next->appendParam(retType);
    result->debug = make_name(to->debug.c_str(), id);
    Params params;
    params.push_back(result);

    // create jump to this new continuation
    size_t csize = args.size() + 1;
    Array<const Def*> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    world().jump(curLambda_, to, cargs);
    curLambda_ = next;

    ++id;

    return result;
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

void BB::eraseEdge(BB* to) {
    to->succs_.erase(this);
    this->preds_.erase(to);
}

World& BB::world() {
    return fct_->world();
}

void BB::emit() {
    switch (succs().size()) {
        case 1:
            world().jump(curLambda_, (*succs().begin())->topLambda(), out_);
            break;
        case 2:
            anydsl_assert(out_.empty(), "sth went wrong with critical edge elimination");
            world().branch(curLambda_, cond_, tbb_->topLambda(), fbb_->topLambda());
            break;
        default: 
            ANYDSL_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

Fct::Fct(World& world, const FctParams& fparams, const Type* retType, const std::string& debug /*= ""*/) 
    : world_(world)
    , retType_(retType)
    , exit_(0)
{
    sealed_ = true;
    fct_ = this;
    curLambda_ = topLambda_ = new Lambda(world.pi0());
    topLambda_->debug = debug;

    for_all (p, fparams) {
        const Param* param = topLambda_->appendParam(p.type);
        in_.push_back(param);
        setVar(p.symbol, param);
        param->debug = p.symbol.str();
    }

    if (retType) {
        retCont_ = topLambda_->appendParam(world.pi1(retType));
        in_.push_back(retCont());
        retCont()->debug = "<return>";
        exit_ = createBB('<' + debug + "_exit>");
    }
}

Fct::~Fct() {
    for_all (bb, cfg_)
        delete bb;
}

BB* Fct::createBB(const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, debug);
    cfg_.insert(bb);

    return bb;
}

void Fct::emit() {
    // exit
    exit()->seal();
    world().jump1(exit_->topLambda_, retCont(), exit()->getVar(Symbol("<result>"), retType())->def);

    world().setLive(exit_->curLambda_);

    for_all (bb, cfg_)
        if (bb != exit())
            bb->emit();

    // fct
    BB::emit();
    world().setReachable(topLambda_);
}

} // namespace anydsl
