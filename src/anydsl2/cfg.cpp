#include "anydsl2/cfg.h"

#include <algorithm>
#include <iterator>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"

#include <iostream>

namespace anydsl2 {

BB::BB(Fct* fct, const std::string& debug) 
    : sealed_(false)
    , visited_(false)
    , fct_(fct)
    , top_(world().lambda())
    , cur_(top_)
{
    top_->debug = debug;
}

BB::~BB() {
    for_all (p, vars_)
        delete p.second;
}

Var* BB::insert(const Symbol& symbol, const Def* def) {
    VarMap::iterator i = vars_.find(symbol);

    if (i != vars_.end()) {
        Var* var = i->second;
        var->store(def);
        return var;
    }

    Var* var = new Var(symbol, def);
    vars_[symbol] = var;

    return var;
}

Var* BB::lookup(const Symbol& symbol, const Type* type) {
    BB::VarMap::iterator i = vars_.find(symbol);

    // if var is known -> return current var
    if (i != vars_.end())
        return i->second;

    // value is undefined
    if (fct_ == this)
        return fct_->lookup_top(symbol, type);

    // insert a 'phi', i.e., create a param and remember to fix the callers
    if (!sealed_ || preds_.size() > 1) {
        const Param* param = top_->append_param(type);
        size_t index = in_.size();
        in_.push_back(param);
        Var* lvar = insert(symbol, param);
        param->debug = symbol.str();

        Todo todo(index, type);

        if (sealed_)
            fix(symbol, todo);
        else
            todos_[symbol] = todo;

        return lvar;
    }

    // unreachable code
    if (preds().empty())
        return insert(symbol, world().bottom(type));
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    Var* lvar = pred->lookup(symbol, type);

    // create copy of lvar in this BB
    return insert(symbol, lvar->load());
}

void BB::seal() {
    assert(!sealed() && "already sealed");
    sealed_ = true;

#ifndef NDEBUG
    if (preds().size() >= 2) {
        for_all (pred, preds_)
            assert(pred->succs().size() <= 1 && "critical edge");
    }
#endif

    for_all (p, todos_)
        fix(p.first, p.second);
}

void BB::fix(const Symbol& symbol, Todo todo) {
    assert(sealed() && "must be sealed");

    size_t index = todo.index();
    const Type* type = todo.type();
    const Param* param = in_[index];
    const Def* same = 0;

    // find Horspool-like phis
    for_all (pred, preds_) {
        const Def* def = pred->lookup(symbol, type)->load();

        if (def->isa<Undef>() || def == param || same == def)
            continue;

        if (same) {
            same = 0;
            goto fix_preds;
        }

        same = def;
    }
    
    if (!same || same == param)
        same = world().bottom(param->type());

    for_all (use, param->copy_uses())
        world().update(use.def(), use.index(), same);

fix_preds:
    for_all (pred, preds_) {
        assert(pred->succs().size() == 1 && "critical edge");
        Out& out = pred->out_;

        // make potentially room for the new arg
        if (index >= out.size())
            out.resize(index + 1);

        assert(!pred->out_[index] && "already set");
        out[index] = same ? same : pred->lookup(symbol, type)->load();
    }

    if (same)
        insert(symbol, same);
}

void BB::jump(BB* to) {
    assert(to);
    this->link(to);
    assert(this->succs().size() == 1 && "wrong number of succs");
}

void BB::branch(const Def* cond, BB* tbb, BB* fbb) {
    assert(tbb);
    assert(fbb);

    cond_ = cond;
    tbb_ = tbb;
    fbb_ = fbb;
    this->link(tbb);
    this->link(fbb);

    assert(succs().size() == 2 && "wrong number of succs");
}

void BB::tail_call(const Def* to, ArrayRef<const Def*> args) {
    Array<const Def*> tcargs(args.size());
    std::copy(args.begin(), args.end(), tcargs.begin());
    cur_->jump(to, tcargs);
}

void BB::return_tail_call(const Def* to, ArrayRef<const Def*> args) {
    Array<const Def*> rargs(args.size() + 1);
    *std::copy(args.begin(), args.end(), rargs.begin()) = fct_->ret();
    cur_->jump(to, rargs);
}

void BB::return_value(const Def* result) { cur_->jump1(fct_->ret(), result); }
void BB::return_void() { cur_->jump0(fct_->ret()); }

const Def* BB::call(const Def* to, ArrayRef<const Def*> args, const Type* rettype) {
    static int id = 0;

    // create next continuation in cascade
    Lambda* next = world().lambda();
    next->debug = cur_->debug + "_" + to->debug;
    const Param* result = next->append_param(rettype);
    result->debug = make_name(to->debug.c_str(), id);

    // create jump to this new continuation
    size_t csize = args.size() + 1;
    Array<const Def*> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    cur_->jump(to, cargs);
    cur_ = next;

    ++id;

    return result;
}

void BB::fixto(BB* to) {
    if (succs().empty())
        this->jump(to);
}

void BB::link(BB* to) {
    BBs::iterator i = succs_.find(to);
    assert(!to->sealed() && "'to' already sealed");

    if (i == succs_.end()) {
        this->succs_.insert(to);
        to->preds_.insert(this);
    } else {
        assert(to->preds_.find(this) != to->preds_.end() && "flow out of sync");
    }
}

World& BB::world() {
    return fct_->world();
}

void BB::emit() {
    size_t num_succs = succs().size();

    if (num_succs == 0) 
        assert(world().lambdas().find(cur_) != world().lambdas().end() && "tail call not finalized");
    else if (num_succs == 1)
        cur_->jump((*succs().begin())->top(), out_);
    else if (num_succs == 2) {
        assert(out_.empty() && "edge is critical");
        cur_->branch(cond_, tbb_->top(), fbb_->top());
    } else
        ANYDSL2_UNREACHABLE;
}

std::string BB::debug() const { return top() ? top()->debug : std::string(); }

//------------------------------------------------------------------------------

Fct::Fct(World& world, ArrayRef<const Type*> types, ArrayRef<Symbol> symbols, 
         size_t return_index, const std::string& debug)
    : world_(world)
    , parent_(0)
{
    assert(types.size() == symbols.size());
    sealed_ = true;
    fct_ = this;
    cur_ = top_ = world.lambda(world.pi(types));
    top_->debug = debug;
    ret_ = (return_index != size_t(-1) ? top()->param(return_index) : 0);

    size_t i = 0;
    for_all (param, top_->params()) {
        Symbol sym = symbols[i++];
        insert(sym, param);
        param->debug = sym.str();
    }
}

Fct::~Fct() {
    for_all (bb, cfg_)
        delete bb;

    for_all (p, letrec_)
        delete p.second;
}

BB* Fct::createBB(const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, debug);
    cfg_.push_back(bb);

    return bb;
}

Var* Fct::lookup_top(const Symbol& symbol, const Type* type) {
    LetRec::const_iterator i = letrec_.find(symbol);
    if (i != letrec_.end())
        return insert(symbol, i->second->top());

    if (parent())
        return parent()->lookup(symbol, type);

    // TODO provide hook instead of fixed functionality
    std::cerr << "'" << symbol << "'" << " may be undefined" << std::endl;
    return insert(symbol, world().bottom(type));
}

void Fct::emit() {
    for_all (bb, cfg_)
        bb->emit();

    BB::emit();
}

void Fct::nest(const Symbol& symbol, Fct* fct) {
    assert(letrec_.find(symbol) == letrec_.end());
    letrec_[symbol] = fct;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
