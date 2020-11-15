#include "thorin/def.h"

#include <algorithm>
#include <stack>

#include "thorin/rewrite.h"
#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/util/utility.h"

namespace thorin {

namespace detail {
    const Def* world_extract(World& world, const Def* def, u64 i, Debug dbg) { return world.extract(def, i, dbg); }
}

/*
 * Def
 */

const char* Def::node_name() const {
    switch (node()) {
#define CODE(op, abbr) case Node::op: return #abbr;
THORIN_NODE(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

Defs Def::extended_ops() const {
    if (isa<Universe>()) return Defs();

    size_t offset = debug() ? 2 : 1;
    return Defs((is_set() ? num_ops_ : 0) + offset, ops_ptr() - offset);
}

size_t Def::num_params() { return param()->type()->lit_arity(); }

const Def* Def::tuple_arity() const {
    if (auto sigma  = isa<Sigma>()) return world().lit_arity(sigma->num_ops());
    if (auto arr    = isa<Arr  >()) return arr->domain();
    if (is_value())                 return type()->tuple_arity();
    assert(is_type());
    return world().lit_arity(1);
}

const Def* Def::arity() const {
    if (auto sigma  = isa<Sigma>()) return world().lit_arity(sigma->num_ops());
    if (auto union_ = isa<Union>()) return world().lit_arity(union_->num_ops());
    if (auto arr    = isa<Arr  >()) return arr->domain();
    if (is_value())                 return type()->arity();
    assert(is_type());
    return world().lit_arity(1);
}

nat_t Def::lit_arity() const { return as_lit<nat_t>(arity()); }
nat_t Def::lit_tuple_arity() const { return as_lit<nat_t>(tuple_arity()); }

bool Def::equal(const Def* other) const {
    if (isa<Universe>() || this->isa_nominal() || other->isa_nominal())
        return this == other;

    bool result = this->node() == other->node() && this->fields() == other->fields() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->op(i) == other->op(i);
    return result;
}

const Def* Def::debug_history() const {
#if THORIN_ENABLE_CHECKS
    auto& w = world();
    if (w.track_history())
        return debug() ? w.insert(debug(), 0_s, w.tuple_str(unique_name())) : w.debug(Name(unique_name()));
#endif
    return debug();
}

std::string Def::name() const     { return debug() ? tuple2str(debug()->out(0)) : std::string{}; }
std::string Def::filename() const { return debug() ? tuple2str(debug()->out(1)) : std::string{}; }
nat_t Def::front_line() const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(0)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::front_col()  const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(1)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_line()  const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(2)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_col()   const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(3)) : std::numeric_limits<nat_t>::max(); }
const Def* Def::meta() const { return debug() ? debug()->out(3) : nullptr; }

std::string Def::loc() const {
    std::ostringstream oss;
    Stream s(oss);
    s.fmt("{}:", filename());

    if (front_col() == nat_t(-1) || back_col() == nat_t(-1)) {
        if (front_line() != back_line())
            s.fmt("{} - {}", front_line(), back_line());
        else
            s.fmt("{}", front_line());
    } else if (front_line() != back_line()) {
        s.fmt("{} col {} - {} col {}", front_line(), front_col(), back_line(), back_col());
    } else if (front_col() != back_col()) {
        s.fmt("{} col {} - {}", front_line(), front_col(), back_col());
    } else {
        s.fmt("{} col {}", front_line(), front_col());
    }

    return oss.str();
}

void Def::set_debug(Debug dbg) const { debug_ = world().debug(dbg); }

void Def::set_name(const std::string& name) const {
    auto& w = world();
    debug_ = debug_ ? w.insert(debug_, 0_s, w.tuple_str(name)) : w.debug(Name(name));
}
void Def::finalize() {
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        if (!op(i)->is_const()) {
            const_ = false;
            const auto& p = op(i)->uses_.emplace(this, i);
            assert_unused(p.second);
        }
        order_ = std::max(order_, op(i)->order_);
    }

    if (!isa<Universe>()) {
        if (!type()->is_const()) {
            const_ = false;
            const auto& p = type()->uses_.emplace(this, -1);
            assert_unused(p.second);
        }
    }

    if (debug()) const_ &= debug()->is_const();
    if (isa<Pi>()) ++order_;
    if (isa<Axiom>()) const_ = true;
}

Def* Def::set(size_t i, const Def* def) {
    if (op(i) == def) return this;
    if (op(i) != nullptr) unset(i);

    if (def != nullptr) {
        assert(i < num_ops() && "index out of bounds");
        ops_ptr()[i] = def;
        order_ = std::max(order_, def->order_);
        const auto& p = def->uses_.emplace(this, i);
        assert_unused(p.second);
    }
    return this;
}

void Def::unset(size_t i) {
    assert(i < num_ops() && "index out of bounds");
    auto def = op(i);
    assert(def->uses_.contains(Use(this, i)));
    def->uses_.erase(Use(this, i));
    assert(!def->uses_.contains(Use(this, i)));
    ops_ptr()[i] = nullptr;
}

bool Def::is_set() const {
    if (!isa_nominal()) {
        assert(std::all_of(ops().begin(), ops().end(), [&](auto op) { return op != nullptr; }) && "structurals must be always set");
        return true;
    }

    if (std::all_of(ops().begin(), ops().end(), [&](auto op) { return op != nullptr; }))
        return true;

    assert(std::all_of(ops().begin(), ops().end(), [&](auto op) { return op == nullptr; }) && "some operands are set, others aren't");
    return false;
}

void Def::make_external() { return world().make_external(this); }
void Def::make_internal() { return world().make_internal(this); }
bool Def::is_external() const { return world().is_external(this); }

std::string Def::unique_name() const { return (isa_nominal() ? std::string{} : std::string{"%"}) + name() + "_" + std::to_string(gid()); }

void Def::replace(Tracker with) const {
    world().DLOG("replace: {} -> {}", this, with);
    assert(type() == with->type());
    assert(!is_replaced());

    if (this != with) {
        for (auto& use : copy_uses()) {
            auto def = const_cast<Def*>(use.def());
            auto index = use.index();
            def->set(index, with);
        }

        uses_.clear();
        substitute_ = with;
    }
}

/*
 * Lam
 */

const Def* Lam::mem_param(thorin::Debug dbg) {
    return thorin::isa<Tag::Mem>(param(0)->type()) ? param(0, dbg) : nullptr;
}

const Def* Lam::ret_param(thorin::Debug dbg) {
    if (num_params() > 0) {
        auto p = param(num_params() - 1, dbg);
        if (auto pi = p->type()->isa<thorin::Pi>(); pi != nullptr && pi->is_cn()) return p;
    }
    return nullptr;
}

bool Lam::is_intrinsic() const { return intrinsic() != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic() && intrinsic() < Intrinsic::_Accelerator_End; }

void Lam::set_intrinsic() {
    // TODO this is slow and inelegant - but we want to remove this code anyway
    auto n = name();
    auto intrin = Intrinsic::None;
    if      (n == "cuda")                 intrin = Intrinsic::CUDA;
    else if (n == "nvvm")                 intrin = Intrinsic::NVVM;
    else if (n == "opencl")               intrin = Intrinsic::OpenCL;
    else if (n == "amdgpu")               intrin = Intrinsic::AMDGPU;
    else if (n == "hls")                  intrin = Intrinsic::HLS;
    else if (n == "parallel")             intrin = Intrinsic::Parallel;
    else if (n == "spawn")                intrin = Intrinsic::Spawn;
    else if (n == "sync")                 intrin = Intrinsic::Sync;
    else if (n == "anydsl_create_graph")  intrin = Intrinsic::CreateGraph;
    else if (n == "anydsl_create_task")   intrin = Intrinsic::CreateTask;
    else if (n == "anydsl_create_edge")   intrin = Intrinsic::CreateEdge;
    else if (n == "anydsl_execute_graph") intrin = Intrinsic::ExecuteGraph;
    else if (n == "vectorize")            intrin = Intrinsic::Vectorize;
    else if (n == "pe_info")              intrin = Intrinsic::PeInfo;
    else if (n == "reserve_shared")       intrin = Intrinsic::Reserve;
    else if (n == "atomic")               intrin = Intrinsic::Atomic;
    else if (n == "cmpxchg")              intrin = Intrinsic::CmpXchg;
    else if (n == "undef")                intrin = Intrinsic::Undef;
    else world().ELOG("unsupported thorin intrinsic");

    set_intrinsic(intrin);
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }

void Lam::app(const Def* callee, const Def* arg, Debug dbg) {
    assert(isa_nominal());
    auto filter = world().lit_false();
    set(filter, world().app(callee, arg, dbg));
}
void Lam::app(const Def* callee, Defs args, Debug dbg) { app(callee, world().tuple(args), dbg); }
void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Debug dbg) {
    return app(world().extract(world().tuple({f, t}), cond, dbg), mem, dbg);
}

void Lam::match(const Def* val, Defs cases, const Def* mem, Debug dbg) {
    return app(world().match(val, cases, dbg), mem, dbg);
}

/*
 * Pi
 */

Pi* Pi::set_domain(Defs domains) { return Def::set(0, world().sigma(domains))->as<Pi>(); }

size_t Pi::num_domains() const { return domain()->lit_arity(); }
const Def* Pi::  domain(size_t i) const { return proj(  domain(), i); }
const Def* Pi::codomain(size_t i) const { return proj(codomain(), i); }

bool Pi::is_returning() const {
    bool ret = false;
    for (auto op : ops()) {
        switch (op->order()) {
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                }
                return false;
            default: continue;
        }
    }
    return ret;
}

/*
 * Ptrn
 */

bool Ptrn::is_trivial() const {
    return matcher()->isa<Param>() && matcher()->as<Param>()->nominal() == this;
}

bool Ptrn::matches(const Def* arg) const {
    return rewrite(as_nominal(), arg, 0) == arg;
}

/*
 * Global
 */

const App* Global::type() const { return thorin::as<Tag::Ptr>(Def::type()); }
const Def* Global::alloced_type() const { return type()->arg(0); }

//------------------------------------------------------------------------------

/*
 * constructors
 */

Def::Def(node_t node, const Def* type, Defs ops, uint64_t fields, const Def* dbg)
    : fields_(fields)
    , node_(unsigned(node))
    , nominal_(false)
    , const_(true)
    , order_(0)
    , num_ops_(ops.size())
    , debug_(dbg)
    , type_(type)
{
    gid_ = world().next_gid();
    std::copy(ops.begin(), ops.end(), ops_ptr());

    if (node == Node::Universe) {
        hash_ = murmur3(gid());
    } else {
        hash_ = hash_combine(hash_begin(node), type->gid(), fields_);
        for (auto op : ops)
            hash_ = hash_combine(hash_, op->gid());
    }
}

Def::Def(node_t node, const Def* type, size_t num_ops, uint64_t fields, const Def* dbg)
    : fields_(fields)
    , node_(node)
    , nominal_(true)
    , const_(false)
    , order_(0)
    , num_ops_(num_ops)
    , debug_(dbg)
    , type_(type)
{
    gid_ = world().next_gid();
    hash_ = murmur3(gid());
    std::fill_n(ops_ptr(), num_ops, nullptr);
    if (!type->is_const()) type->uses_.emplace(this, -1);
}

Axiom::Axiom(NormalizeFn normalizer, const Def* type, u32 tag, u32 flags, const Def* dbg)
    : Def(Node, type, Defs{}, (nat_t(tag) << 32_u64) | nat_t(flags), dbg)
{
    u16 currying_depth = 0;
    while (auto pi = type->isa<Pi>()) {
        ++currying_depth;
        type = pi->codomain();
    }

    normalizer_depth_.set(normalizer, currying_depth);
}

Kind::Kind(World& world, Tag tag)
    : Def(Node, tag == Star  ? (const Def*) world.universe() :
                tag == Multi ? (const Def*) world.kind(Star) :
                               (const Def*) world.kind(Multi), Defs{}, fields_t(tag), nullptr)
{}

Nat::Nat(World& world)
    : Def(Node, world.kind(Kind::Star), Defs{}, 0, nullptr)
{}

/*
 * param
 */

const Param* Def::param(Debug dbg) {
    if (auto lam    = isa<Lam  >()) return world().param(lam ->domain(), lam,    dbg);
    if (auto ptrn   = isa<Ptrn >()) return world().param(ptrn->domain(), ptrn,   dbg);
    if (auto pi     = isa<Pi   >()) return world().param(pi  ->domain(), pi,     dbg);
    if (auto sigma  = isa<Sigma>()) return world().param(sigma,          sigma,  dbg);
    if (auto union_ = isa<Union>()) return world().param(union_,         union_, dbg);
    THORIN_UNREACHABLE;
}

const Param* Def::param() { return param(Debug()); }
const Def*   Def::param(size_t i) { return param(i, Debug()); }

/*
 * apply/reduce
 */

Array<const Def*> Def::apply(const Def* arg) const {
    if (auto nom = isa_nominal()) return nom->apply(arg);
    return ops();
}

Array<const Def*> Def::apply(const Def* arg) {
    auto& cache = world().data_.cache_;
    if (auto res = cache.lookup({this, arg})) return *res;

    return cache[{this, arg}] = rewrite(this, arg);
}

const Def* Def::reduce() const {
    auto def = this;
    while (auto app = def->isa<App>()) {
        auto callee = app->callee()->reduce();
        if (callee->isa_nominal()) {
            def = callee->apply(app->arg()).back();
        } else {
            def = callee != app->callee() ? world().app(callee, app->arg(), app->debug()) : app;
            break;
        }
    }
    return def;
}

const Def* Def::refine(size_t i, const Def* new_op) const {
    Array<const Def*> new_ops(ops());
    new_ops[i] = new_op;
    return rebuild(world(), type(), new_ops, debug());
}

/*
 * rebuild
 */

const Def* App     ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.app(o[0], o[1], dbg); }
const Def* Arr     ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.arr(o[0], o[1], dbg); }
const Def* Axiom   ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.axiom(normalizer(), t, tag(), flags(), dbg); }
const Def* Bot     ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.bot(t, dbg); }
const Def* CPS2DS  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.cps2ds(o[0], dbg); }
const Def* Case    ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.case_(o[0], o[1], dbg); }
const Def* DS2CPS  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.ds2cps(o[0], dbg); }
const Def* Extract ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.extract(t, o[0], o[1], dbg); }
const Def* Global  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.global(o[0], o[1], is_mutable(), dbg); }
const Def* Insert  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.insert(o[0], o[1], o[2], dbg); }
const Def* Kind    ::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.kind(as<Kind>()->tag()); }
const Def* Lam     ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.lam(t->as<Pi>(), o[0], o[1], dbg); }
const Def* Lit     ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.lit(t, get(), dbg); }
const Def* Match   ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.match(o[0], o.skip_front(), dbg); }
const Def* Nat     ::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.type_nat(); }
const Def* Pack    ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.pack(t->arity(), o[0], dbg); }
const Def* Param   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.param(t, o[0]->as_nominal(), dbg); }
const Def* Pi      ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.pi(o[0], o[1], dbg); }
const Def* Proxy   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.proxy(t, o, as<Proxy>()->index(), as<Proxy>()->flags(), dbg); }
const Def* Sigma   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.sigma(t, o, dbg); }
const Def* Succ    ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.succ(t, tuplefy(), dbg); }
const Def* Top     ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.top(t, dbg); }
const Def* Tuple   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.tuple(t, o, dbg); }
const Def* Union   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.union_(t, o, dbg); }
const Def* Universe::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.universe(); }
const Def* Which   ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.which(o[0], dbg); }

/*
 * stub
 */

Lam*   Lam  ::stub(World& w, const Def* t, const Def* dbg) { return w.lam(t->as<Pi>(), cc(), intrinsic(), dbg); }
Pi*    Pi   ::stub(World& w, const Def* t, const Def* dbg) { return w.pi(t, Debug{dbg}); }
Ptrn*  Ptrn ::stub(World& w, const Def* t, const Def* dbg) { return w.ptrn(t->as<Case>(), dbg); }
Sigma* Sigma::stub(World& w, const Def* t, const Def* dbg) { return w.sigma(t, num_ops(), dbg); }
Union* Union::stub(World& w, const Def* t, const Def* dbg) { return w.union_(t, num_ops(), dbg); }

/*
 * is_value
 */

bool Universe::is_value() const { return false; }
bool Kind    ::is_value() const { return false; }
bool Arr     ::is_value() const { return false; }
bool Case    ::is_value() const { return false; }
bool Nat     ::is_value() const { return false; }
bool Pi      ::is_value() const { return false; }
bool Sigma   ::is_value() const { return false; }
bool Union   ::is_value() const { return false; }
bool Global  ::is_value() const { return true; }
bool Insert  ::is_value() const { return true; }
bool Lam     ::is_value() const { return true; }
bool Pack    ::is_value() const { return true; }
bool Ptrn    ::is_value() const { return true; }
bool Tuple   ::is_value() const { return true; }
bool Which   ::is_value() const { return true; }
bool Succ    ::is_value() const { return as<Succ>()->tuplefy(); }

/*
 * is_type
 */

bool Universe::is_type() const { return false; }
bool Kind    ::is_type() const { return false; }
bool Arr     ::is_type() const { return true; }
bool Case    ::is_type() const { return true; }
bool Nat     ::is_type() const { return true; }
bool Pi      ::is_type() const { return true; }
bool Sigma   ::is_type() const { return true; }
bool Union   ::is_type() const { return true; }
bool Global  ::is_type() const { return false; }
bool Insert  ::is_type() const { return false; }
bool Lam     ::is_type() const { return false; }
bool Pack    ::is_type() const { return false; }
bool Ptrn    ::is_type() const { return false; }
bool Tuple   ::is_type() const { return false; }
bool Which   ::is_type() const { return false; }
bool Succ    ::is_type() const { return as<Succ>()->sigmafy(); }

/*
 * is_kind
 */

bool Kind    ::is_kind() const { return true; }
bool Universe::is_kind() const { return false; }

}
