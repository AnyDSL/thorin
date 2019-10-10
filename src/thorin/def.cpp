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

const Def* Def::arity() const {
    if (auto sigma  = isa<Sigma>()) return world().lit_arity(sigma->num_ops());
    if (auto union_ = isa<Union>()) return world().lit_arity(union_->num_ops());
    if (auto arr    = isa<Arr  >()) return arr->domain();
    return world().lit_arity(1);
}

nat_t Def::lit_arity() const {
    if (auto sigma  = isa<Sigma>()) return sigma->num_ops();
    if (auto union_ = isa<Union>()) return union_->num_ops();
    if (auto arr    = isa<Arr  >()) return as_lit<nat_t>(arr->domain());
    return 1;
}

bool Def::equal(const Def* other) const {
    if (this->isa_nominal() || other->isa_nominal())
        return this == other;

    bool result = this->node() == other->node() && this->fields() == other->fields() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->op(i) == other->op(i);
    return result;
}


// TODO
const Def* Def::debug_history() const {
//#if THORIN_ENABLE_CHECKS
    //return world().track_history() ? Debug(loc(), world().tuple_str(unique_name())) : debug();
//#else
    return debug();
//#endif
}

std::string Def::name() const     { return debug() ? tuple2str(debug()->out(0)) : std::string{}; }
std::string Def::filename() const { return debug() ? tuple2str(debug()->out(1)) : std::string{}; }
nat_t Def::front_line() const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(0)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::front_col()  const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(1)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_line()  const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(2)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_col()   const { return debug() ? as_lit<nat_t>(debug()->out(2)->out(3)) : std::numeric_limits<nat_t>::max(); }
const Def* Def::meta() const { return debug() ? debug()->out(3) : nullptr; }

const char* Def::node_name() const {
    switch (node()) {
#define CODE(op, abbr) case Node::op: return #abbr;
THORIN_NODE(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

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

void Def::finalize() {
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        auto o = op(i);
        assert(o != nullptr);
        const_ |= o->is_const();
        order_ = std::max(order_, o->order_);
        const auto& p = o->uses_.emplace(this, i);
        assert_unused(p.second);
    }

    const_ |= type()->is_const();
    if (isa<Pi>()) ++order_;
}

Def* Def::set(size_t i, const Def* def) {
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

std::string Def::unique_name() const { return name() + "_" + std::to_string(gid()); }

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
    return param(0)->type()->isa<Mem>() ? param(0, dbg) : nullptr;
}

const Def* Lam::ret_param(thorin::Debug dbg) {
    auto p = param(num_params() - 1, dbg);
    assert(p->type()->as<thorin::Pi>()->is_cn());
    return p;
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

void Lam::match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Debug dbg) {
    Array<const Def*> args(patterns.size() + 2);

    args[0] = val;
    args[1] = otherwise;
    assert(patterns.size() == lams.size());
    for (size_t i = 0; i < patterns.size(); i++)
        args[i + 2] = world().tuple({patterns[i], lams[i]}, dbg);

    return app(world().match(val->type(), patterns.size()), args, dbg);
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

const Def* Pi::apply(const Def* arg) const {
    if (auto pi = isa_nominal<Pi>()) return rewrite(pi, arg);
    return codomain();
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

Def::Def(node_t node, RebuildFn rebuild, const Def* type, Defs ops, uint64_t fields, const Def* dbg)
    : type_(type)
    , rebuild_(rebuild)
    , debug_(dbg)
    , fields_(fields)
    , node_((unsigned)node)
    , nominal_(false)
    , const_(node != Node::Param)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(ops.size())
{
    type->uses_.emplace(this, -1);
    std::copy(ops.begin(), ops.end(), ops_ptr());
    hash_ = hash_combine(hash_begin(node), type->gid(), fields_);
    for (auto op : ops)
        hash_ = hash_combine(hash_, op->gid());
}

Def::Def(node_t node, StubFn stub, const Def* type, size_t num_ops, uint64_t fields, const Def* dbg)
    : type_(type)
    , stub_(stub)
    , debug_(dbg)
    , fields_(fields)
    , node_(node)
    , nominal_(true)
    , const_(false)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(num_ops)
    , hash_(murmur3(gid()))
{
    if (node != Node::Universe) type->uses_.emplace(this, -1);
    std::fill_n(ops_ptr(), num_ops, nullptr);
}

Axiom::Axiom(NormalizeFn normalizer, const Def* type, u32 tag, u32 flags, const Def* dbg)
    : Def(Node, rebuild, type, Defs{}, (nat_t(tag) << 32_u64) | nat_t(flags), dbg)
{
    u16 currying_depth = 0;
    while (auto pi = type->isa<Pi>()) {
        ++currying_depth;
        type = pi->codomain();
    }

    normalizer_depth_.set(normalizer, currying_depth);
}

KindArity::KindArity(World& world)
    : Def(Node, rebuild, world.kind_multi(), Defs{}, 0, nullptr)
{}

KindMulti::KindMulti(World& world)
    : Def(Node, rebuild, world.kind_star(), Defs{}, 0, nullptr)
{}

KindStar::KindStar(World& world)
    : Def(Node, rebuild, world.universe(), Defs{}, 0, nullptr)
{}

Nat::Nat(World& world)
    : Def(Node, rebuild, world.kind_star(), Defs{}, 0, nullptr)
{}

Mem::Mem(World& world)
    : Def(Node, rebuild, world.kind_star(), Defs{}, 0, nullptr)
{}

const Param* Def::param(Debug dbg) {
    if (auto lam   = isa<Lam  >()) return world().param(lam ->domain(), lam,   dbg);
    if (auto pi    = isa<Pi   >()) return world().param(pi  ->domain(), pi,    dbg);
    if (auto sigma = isa<Sigma>()) return world().param(sigma,          sigma, dbg);
    THORIN_UNREACHABLE;
}

size_t Def::num_params() { return param()->type()->lit_arity(); }

/*
 * rebuild
 */

const Def* Axiom      ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.axiom(d->as<Axiom>()->normalizer(), t, d->as<Axiom>()->tag(), d->as<Axiom>()->flags(), dbg); }
const Def* Lam        ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.lam(t->as<Pi>(), o[0], o[1], dbg); }
const Def* CPS2DS     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.cps2ds(o[0], dbg); }
const Def* DS2CPS     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.ds2cps(o[0], dbg); }
const Def* Sigma      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.sigma(t, o, dbg); }
const Def* Union      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.union_(t, o, dbg); }
const Def* Analyze    ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { return w.analyze(t, o, d->fields(), dbg); }
const Def* App        ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.app(o[0], o[1], dbg); }
const Def* Bot        ::rebuild(const Def*  , World& w, const Def* t, Defs  , const Def* dbg) { return w.bot(t, dbg); }
const Def* Top        ::rebuild(const Def*  , World& w, const Def* t, Defs  , const Def* dbg) { return w.top(t, dbg); }
const Def* Extract    ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.extract(o[0], o[1], dbg); }
const Def* Global     ::rebuild(const Def* d, World& w, const Def*  , Defs o, const Def* dbg) { return w.global(o[0], o[1], d->as<Global>()->is_mutable(), dbg); }
const Def* Insert     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.insert(o[0], o[1], o[2], dbg); }
const Def* Match_     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.match_(o[0], o.skip_front(), dbg); }
const Def* KindArity  ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_arity(); }
const Def* KindMulti  ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_multi(); }
const Def* KindStar   ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_star(); }
const Def* Lit        ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.lit(t, as_lit<nat_t>(d), dbg); }
const Def* Nat        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_nat(); }
const Def* Mem        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_mem(); }
const Def* Pack       ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.pack(t->arity(), o[0], dbg); }
const Def* Param      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.param(t, o[0]->as_nominal(), dbg); }
const Def* Pi         ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.pi(o[0], o[1], dbg); }
const Def* Tuple      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.tuple(t, o, dbg); }
const Def* Variant_   ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.variant_(t, o[0], o[1], dbg); }
const Def* Arr        ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.arr(o[0], o[1], dbg); }
const Def* Variant    ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.variant(t->as<VariantType>(), o[0], dbg); }
const Def* VariantType::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.variant_type(o, dbg); }
const Def* Succ       ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.succ(t, d->as<Succ>()->tuplefy(), dbg); }

/*
 * stub
 */

Def* Universe::stub(const Def*  , World& w, const Def*  , const Def*    ) { return const_cast<Universe*>(w.universe()); }
Def* Lam     ::stub(const Def* d, World& w, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return w.lam(t->as<Pi>(), d->as<Lam>()->cc(), d->as<Lam>()->intrinsic(), dbg); }
Def* Pi      ::stub(const Def* d, World& w, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return w.pi(t, Debug{dbg}); }
Def* Sigma   ::stub(const Def* d, World& w, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return w.sigma(t, d->num_ops(), dbg); }
Def* Union   ::stub(const Def* d, World& w, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return w.union_(t, d->num_ops(), dbg); }

template void Streamable<Def>::dump() const;

}
