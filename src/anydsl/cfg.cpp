#include "anydsl/cfg.h"

#include "anydsl/literal.h"
#include "anydsl/jump.h"
#include "anydsl/world.h"
#include "anydsl/binding.h"
#include "anydsl/util/foreach.h"

using namespace anydsl;

namespace anydsl {

//------------------------------------------------------------------------------

BB::BB(BB* parent, const Pi* pi, const std::string& name) 
    : lambda_(pi->world().createLambda(pi))
    , visited_(false)
    , fct_(0)
{
    lambda_->debug = name;
}

BB::BB(Fct* fct, World& world, const std::string& name) 
    : lambda_(world.createLambda(0))
    , visited_(false)
    , fct_(fct)
{
    lambda_->debug = name;
}

World& BB::world() { 
    return lambda_->world();
}

/*static*/ BB* BB::createBB(Fct* fct, World& world, const std::string& name) {
    return new BB(fct, world, name);
}

void BB::goesto(BB* to) {
    assert(to);
    world().createJump(this->lambda(), to->lambda());
    this->flowsto(to);
    anydsl_assert(this->succ().size() == 1, "wrong number of succ");
}

void BB::branches(Def* cond, BB* tbb, BB* fbb) {
    assert(tbb);
    assert(fbb);
    world().createBranch(lambda(), cond, tbb->lambda(), fbb->lambda());
    this->flowsto(tbb);
    this->flowsto(fbb);
    anydsl_assert(succ().size() == 2, "wrong number of succ");
}

void BB::invokes(Def* fct) {
    anydsl_assert(fct, "must be valid");
    world().createJump(this->lambda(), fct);
    anydsl_assert(this->succ().size() == 0, "wrong number of succ");
    // succs by invokes are not captured in the CFG
}

void BB::fixto(BB* to) {
    assert(!lambda()->jump());
    this->goesto(to);
}


void BB::flowsto(BB* to) {
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
        ParamIter x = i.second;
        Symbol sym = i.first;

        FOREACH(pred, pred_) {
            anydsl_assert(!pred->succ_.empty(), "must have at least one succ");
            anydsl_assert(pred->succ_.find(this) != pred->succ_.end(), "incorrectly wired");
            pred->finalize(x, sym);
        }
    }
}

void BB::finalize(ParamIter param, const Symbol sym) {
    if (Beta* beta = getBeta()) {
        //anydsl_assert(beta->args().empty(), "must be empty");
        fixBeta(beta, param, sym, 0);
    } else if (Branch* branch = getBranch()) {
        Lambda* lam[2] = {scast<Lambda>(branch-> trueExpr.def()), 
                          scast<Lambda>(branch->falseExpr.def()) };

        for (size_t i = 0; i < 2; ++i) {
            Beta* beta = scast<Beta>(lam[i]->body());
            fixBeta(beta, param, sym, 0);
        }
    } else
        ANYDSL_UNREACHABLE;
}

void BB::fixBeta(Beta* beta, ParamIter param, const Symbol sym, Type* type) {
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
#if 0
                FOREACH(pred, pred_)
                    pred->finalize(param, sym);
#endif
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

void BB::dfs(BBList& bbs) {
    visited_ = true;

    FOREACH(bb, succ())
        if (!bb->visited_)
            dfs(bbs);

    bbs.push_back(this);
}

//------------------------------------------------------------------------------

Fct::Fct(const Pi* pi, const Symbol sym)
    : BB(0, pi, sym.str())
    , exit_(0)
    , retParam_(0)
{}

Fct::Fct(World& world, const Symbol sym)
    : BB(0, world.pi0(), sym.str())
    , exit_(0)
    , retParam_(0)
{}

/*static*/ BB* Fct::createBB(const std::string& name /*= ""*/) {
    BB* bb = BB::createBB(this, world(), name);
    cfg_.insert(bb);

    return bb;
}

void Fct::setReturnCont(const Type* retType) {
    anydsl_assert(!exit_, "already set");

    Symbol resSymbol = "<result>";
    retParam_ = *lambda_->appendParam(world().pi1(retType));
    retParam_->debug = "<result>";
    setVN(new Binding(resSymbol, world().undef(retType)));
    exit_ = createBB("<exit>");
    exit_->invokes(retParam_);
    exit_->lambda()->jump()->ops_append(getVN(resSymbol, retType, false)->def);
}

void Fct::insertReturnStmt(BB* bb, Def* def) {
    anydsl_assert(bb, "must be valid");
    bb->goesto(exit_);
    bb->lambda()->jump()->ops_append(def);
}

Binding* Fct::getVN(const Symbol sym, const Type* type, bool finalize) {
    BB::ValueMap::iterator i = values_.find(sym);
    if (i == values_.end()) {
        Undef* undef = world().undef(type);
        std::cerr << "may be undefined: " << sym << std::endl;

        return new Binding(sym, undef);
    }

    anydsl_assert(i->second->def, "must be valid");
    return i->second;
}


void Fct::buildDomTree() {
    dfs(postorder_);
    anydsl_assert(postorder_.back() == this, "last node must be start node, i.e., 'this'");
    size_t last = postorder_.size() - 1;

    idoms_.resize(postorder_.size());

    // init idoms to 0, set visited_ to false
    for (size_t i = last - 1; i >= 0; --i) {
        BB* bb = postorder_[i];
        idoms_[i] = 0;
        bb->poIndex_ = i;
    }

    idoms_.back() = this;

    bool changed = true;
    while (changed) {
        // for each bb in reverse post-order except start node
        for (size_t bb_i = last - 1; bb_i >= 0; --bb_i) {
            BB* bb = postorder_[bb_i];
            bb->visited_ = true;

            BB* new_bb = 0;
            // find processed pred of bb
            FOREACH(pred, bb->pred()) {
                if (pred->poIndex_ > bb_i) {
                    new_bb = pred;
                    break;
                }
            }
            anydsl_assert(new_bb, "no processed pred of bb found");
            size_t new_i = new_bb->poIndex_;

            // for all un processed preds of bb
            FOREACH(pred, bb->pred()) {
                size_t pred_i = pred->poIndex_;
                if (!pred->visited_) {
                    if (idoms_[pred_i])
                        new_i = intersect(pred_i, new_i);
                }
            }

            if (idoms_[bb_i] != new_bb) {
                idoms_[bb_i] = new_bb;
                changed = true;
            }
        }
    }

    // now build dom tree
    for (size_t i = last - 1; i >= 0; --i) {
        BB* bb = postorder_[i];
        BB* dom = idoms_[i];
        dom->lambda()->insert(bb->lambda());
    }
}

size_t Fct::intersect(size_t i, size_t j) {
    while (i != j) {
        while (i < j) 
            i = idoms_[i]->poIndex_;
        while (j < i) 
            j = idoms_[j]->poIndex_;
    }
    return i;
}

//------------------------------------------------------------------------------

} // namespace impala
