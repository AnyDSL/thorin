#include "anydsl/cfg.h"

#include <algorithm>
#include <iterator>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

namespace anydsl2 {

BB::BB(Fct* fct, const std::string& debug) 
    : sealed_(false)
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
    if (fct_ == this) {
        // TODO provide hook instead of fixed functionality
        std::cerr << "'" << symbol << "'" << " may be undefined" << std::endl;
        return insert(symbol, world().bottom(type));
    }

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

void BB::return_call(const Def* to, ArrayRef<const Def*> args) {
    Array<const Def*> rargs(args.size() + 1);
    *std::copy(args.begin(), args.end(), rargs.begin()) = fct_->ret();
    cur_->jump(to, rargs);
}

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
        ANYDSL_UNREACHABLE;
}

//------------------------------------------------------------------------------

Fct::Fct(World& world, 
         ArrayRef<const Type*> types, ArrayRef<Symbol> symbols, 
         const Type* rettype, const std::string& debug)
    : world_(world)
    , rettype_(rettype)
    , exit_(0)
{
    assert(types.size() == symbols.size());
    sealed_ = true;
    fct_ = this;
    cur_ = top_ = world.lambda(world.pi(types), Lambda::Extern);
    top_->debug = debug;

    size_t i = 0;
    for_all (param, top_->params()) {
        Symbol sym = symbols[i++];
        insert(sym, param);
        param->debug = sym.str();
    }

    if (rettype_) {
        ret_ = top_->append_param(world.pi1(rettype));
        ret()->debug = "<return>";
        exit_ = createBB('<' + debug + "_exit>");
    }
}

Fct::~Fct() {
    for_all (bb, cfg_)
        delete bb;
}

BB* Fct::createBB(const std::string& debug /*= ""*/) {
    BB* bb = new BB(this, debug);
    cfg_.push_back(bb);

    return bb;
}

void Fct::emit() {
    if (rettype()) {
        exit()->seal();
        exit_->top_->jump1(ret(), exit()->lookup(Symbol("<result>"), rettype())->load());
    }

    for_all (bb, cfg_)
        bb->emit();

    BB::emit();
}

//------------------------------------------------------------------------------

} // namespace anydsl2
