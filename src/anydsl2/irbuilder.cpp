#include "anydsl2/irbuilder.h"

#include <algorithm>
#include <iterator>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"

#include <iostream>

namespace anydsl2 {

BB::BB(Fct* fct, const std::string& name) 
    : sealed_(false)
    , visited_(false)
    , fct_(fct)
    , top_(world().lambda(name))
    , cur_(top_)
{}

Var* BB::set_value(size_t handle, const Def* def) {
    if (Var* var = vars_.find(handle))
        return var;

    Var* var = new Var(handle, def);
    vars_[handle] = var;
    return var;
}

Var* BB::get_value(size_t handle, const Type* type, const std::string& name) {
    if (Var* var = vars_.find(handle))
        return var;

    // value is undefined
    if (fct_ == this)
        return fct_->get_value_top(handle, type, name);

    // insert a 'phi', i.e., create a param and remember to fix the callers
    if (!sealed_ || preds_.size() > 1) {
        const Param* param = top_->append_param(type, name);
        size_t index = in_.size();
        in_.push_back(param);
        Var* lvar = set_value(handle, param);

        Todo todo(handle, index, type);

        if (sealed_)
            fix(todo);
        else
            todos_.push_back(todo);

        return lvar;
    }

    // unreachable code
    if (preds().empty())
        return set_value(handle, world().bottom(type));
    
    // look in pred if there exists exactly one pred
    assert(preds().size() == 1);

    BB* pred = *preds().begin();
    Var* lvar = pred->get_value(handle, type);

    // create copy of lvar in this BB
    return set_value(handle, lvar->load());
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

    for_all (todo, todos_)
        fix(todo);
}

void BB::fix(Todo todo) {
    assert(sealed() && "must be sealed");

    size_t handle = todo.handle();
    size_t index = todo.index();
    const Type* type = todo.type();
    const Param* param = in_[index];
    const Def* same = 0;

    // find Horspool-like phis
    for_all (pred, preds_) {
        const Def* def = pred->get_value(handle, type)->load();

        if (def->isa<Undef>() || def == param || same == def)
            continue;

        if (same) {
            same = 0;
            goto fix_preds;
        }
        same = def;
    }
    
goto fix_preds; // HACK fix cond_
    if (!same || same == param)
        same = world().bottom(param->type());

    for_all (use, param->uses())
        world().update(use.def(), use.index(), same);

fix_preds:
    for_all (pred, preds_) {
        assert(pred->succs().size() == 1 && "critical edge");
        Out& out = pred->out_;

        // make potentially room for the new arg
        if (index >= out.size())
            out.resize(index + 1);

        assert(!pred->out_[index] && "already set");
        out[index] = same ? same : pred->get_value(handle, type)->load();
    }

    if (same)
        set_value(handle, same);
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
    Lambda* next = world().lambda(cur_->name + "_" + to->name);
    const Param* result = next->append_param(rettype, make_name(to->name.c_str(), id));

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
    } else
        assert(to->preds_.find(this) != to->preds_.end() && "flow out of sync");
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

std::string BB::name() const { return top() ? top()->name : std::string(); }

//------------------------------------------------------------------------------

Fct::Fct(World& world, const Pi* pi, ArrayRef<size_t> handles, ArrayRef<Symbol> symbols, 
         size_t return_index, const std::string& name)
    : world_(world)
    , parent_(0)
{
    size_t num = pi->size();
    assert(pi->size() == num);
    assert(handles.size() == num);
    assert(symbols.size() == num);
    sealed_ = true;
    fct_ = this;
    cur_ = top_ = world.lambda(pi);
    top_->name = name;
    ret_ = (return_index != size_t(-1) ? top()->param(return_index) : 0);

    for (size_t i = 0; i < num; ++i) {
        set_value(handles[i], top_->param(i));
        top_->param(i)->name = symbols[i].str();
    }
}

Fct::~Fct() {
    for_all (bb, cfg_)
        delete bb;
}

BB* Fct::createBB(const std::string& name /*= ""*/) {
    BB* bb = new BB(this, name);
    cfg_.push_back(bb);

    return bb;
}

Var* Fct::get_value_top(size_t handle, const Type* type, const std::string& name) {
    if (parent())
        return parent()->get_value(handle, type, name);

    // TODO provide hook instead of fixed functionality
    std::cerr << "'" << name << "'" << " may be undefined" << std::endl;
    return set_value(handle, world().bottom(type));
}

void Fct::emit() {
    for_all (bb, cfg_)
        bb->emit();

    BB::emit();
}

//------------------------------------------------------------------------------

} // namespace anydsl2
