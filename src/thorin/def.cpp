#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/primop.h"
#include "thorin/rewrite.h"
#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/util/log.h"
#include "thorin/util/utility.h"

namespace thorin {

namespace detail {
    const Def* world_extract(World& world, const Def* def, u64 i) { return world.extract(def, i); }
}

/*
 * Def
 */

size_t Def::num_outs() const { return as_lit<u64>(arity()); }
const App* Def::decurry() const { return as<App>()->callee()->as<App>(); }

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
nat_t Def::front_line() const { return debug() ? as_lit<nat_t>(debug()->out(2)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::front_col()  const { return debug() ? as_lit<nat_t>(debug()->out(3)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_line()  const { return debug() ? as_lit<nat_t>(debug()->out(4)) : std::numeric_limits<nat_t>::max(); }
nat_t Def::back_col()   const { return debug() ? as_lit<nat_t>(debug()->out(5)) : std::numeric_limits<nat_t>::max(); }

std::string Def::loc() const {
    std::ostringstream os;
    os << filename() << ':';

    if (front_col() == nat_t(-1) || back_col() == nat_t(-1)) {
        if (front_line() != back_line())
            streamf(os, "{} - {}", front_line(), back_line());
        else
            streamf(os, "{}", front_line());
    } else if (front_line() != back_line()) {
        streamf(os, "{} col {} - {} col {}", front_line(), front_col(), back_line(), back_col());
    } else if (front_col() != back_col()) {
        streamf(os, "{} col {} - {}", front_line(), front_col(), back_col());
    } else {
        streamf(os, "{} col {}", front_line(), front_col());
    }

    return os.str();
}

void Def::finalize() {
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        auto o = op(i);
        assert(o != nullptr);
        order_ = std::max(order_, o->order_);
        const auto& p = o->uses_.emplace(this, i);
        assert_unused(p.second);
    }

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

std::string Def::unique_name() const {
    std::ostringstream oss;
    if (!name().empty()) oss << name();
    oss <<  '_' << gid();
    return oss.str();
}

void Def::replace(Tracker with) const {
    DLOG("replace: {} -> {}", this, with);
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

Def::Sort Def::sort() const {
    if (node()                 == Node::Universe) return Sort::Universe;
    if (type()->node()         == Node::Universe) return Sort::Kind;
    if (type()->type()->node() == Node::Universe) return Sort::Type;
    assert(type()->type()->type()->node() == Node::Universe);
    return Sort::Term;
}

void Def::dump() const {
    if (!isa_nominal() && num_ops() > 1)
        stream_assignment(std::cout);
    else {
        std::cout << this;
        std::cout << std::endl;
    }
}

/*
 * Lam
 */

const Def* Lam::mem_param() const {
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (p->type()->isa<Mem>())
            return p;
    }
    return nullptr;
}

const Def* Lam::ret_param() const {
    const Def* result = nullptr;
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (p->type()->order() >= 1) {
            assertf(result == nullptr, "only one ret_param allowed");
            result = p;
        }
    }
    return result;
}

Lams Lam::preds() const {
    std::vector<Lam*> preds;
    std::queue<Use> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        for (auto use : def->uses()) {
            if (done.find(use) == done.end()) {
                queue.push(use);
                done.insert(use);
            }
        }
    };

    done.insert(this);
    enqueue(this);

    while (!queue.empty()) {
        auto use = pop(queue);
        if (auto lam = use->isa_nominal<Lam>()) {
            preds.push_back(lam);
            continue;
        }

        enqueue(use);
    }

    return preds;
}

Lams Lam::succs() const {
    std::vector<Lam*> succs;
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(this);
    enqueue(body());

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto lam = def->isa_nominal<Lam>()) {
            succs.push_back(lam);
            continue;
        }

        for (auto op : def->ops())
            enqueue(op);
    }

    return succs;
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
    else ELOG("unsupported thorin intrinsic");

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
void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Debug dbg) { return app(world().op_select(cond, t, f, dbg), mem, dbg); }

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

Array<const Def*> Pi::domains() const {
    size_t n = num_domains();
    Array<const Def*> domains(n);
    for (size_t i = 0; i != n; ++i)
        domains[i] = domain(i);
    return domains;
}

size_t Pi::num_domains() const {
    if (auto sigma = domain()->isa<Sigma>())
        return sigma->num_ops();
    return 1;
}

const Def* Pi::domain(size_t i) const {
    if (auto sigma = domain()->isa<Sigma>())
        return sigma->op(i);
    return domain();
}

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

const Def* Pi::apply(const Def* arg) const { return isa_nominal() ? rewrite(codomain(), param(), arg) : codomain(); }

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
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(ops.size())
{
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
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(num_ops)
    , hash_(murmur3(gid()))
{
    std::fill_n(ops_ptr(), num_ops, nullptr);
}

Axiom::Axiom(NormalizeFn normalizer, const Def* type, u32 tag, u32 flags, const Def* dbg)
    : Def(Node, stub, type, 0, (nat_t(tag) << 32_u64) | nat_t(flags), dbg)
{
    u16 currying_depth = 0;
    while (auto pi = type->isa<Pi>()) {
        ++currying_depth;
        type = pi->codomain();
    }

    normalizer_depth_.set(normalizer, currying_depth);
}

KindArity::KindArity(World& world)
    : Def(Node, rebuild, world.universe(), Defs{}, 0, nullptr)
{}

KindMulti::KindMulti(World& world)
    : Def(Node, rebuild, world.universe(), Defs{}, 0, nullptr)
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

/*
 * param
 */

const Param* Lam::param(Debug dbg) const { return world().param(domain(), as_nominal<Lam>(), dbg); }
const Param* Pi ::param(Debug dbg) const { return world().param(domain(), as_nominal<Pi >(), dbg); }

/*
 * arity
 */

const Def* Def  ::arity() const { return is_term() ? type()->arity() : world().lit_arity_1(); }
const Def* Sigma::arity() const { return world().lit_arity(num_ops()); }
nat_t Def::lit_arity() const { return as_lit<nat_t>(arity()); }

/*
 * equal
 */

bool Def::equal(const Def* other) const {
    if (this->isa_nominal() || other->isa_nominal())
        return this == other;

    bool result = this->node() == other->node() && this->fields() == other->fields() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->op(i) == other->op(i);
    return result;
}

/*
 * rebuild
 */

const Def* Lam        ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { assert(!d->isa_nominal()); return w.lam(t->as<Pi>(), o[0], o[1], dbg); }
const Def* Sigma      ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { assert(!d->isa_nominal()); return w.sigma(t, o, dbg); }
const Def* Analyze    ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.analyze(t, o, dbg); }
const Def* App        ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.app(o[0], o[1], dbg); }
const Def* Bot        ::rebuild(const Def*  , World& w, const Def* t, Defs  , const Def* dbg) { return w.bot(t, dbg); }
const Def* Top        ::rebuild(const Def*  , World& w, const Def* t, Defs  , const Def* dbg) { return w.top(t, dbg); }
const Def* Extract    ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.extract(o[0], o[1], dbg); }
const Def* Insert     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.insert(o[0], o[1], o[2], dbg); }
const Def* KindArity  ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_arity(); }
const Def* KindMulti  ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_multi(); }
const Def* KindStar   ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.kind_star(); }
const Def* Lit        ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.lit(t, as_lit<nat_t>(d), dbg); }
const Def* Nat        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_nat(); }
const Def* Mem        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_mem(); }
const Def* Pack       ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.pack(t->arity(), o[0], dbg); }
const Def* Param      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.param(t, o[0]->as_nominal(), dbg); }
const Def* Pi         ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.pi(o[0], o[1], dbg); }
const Def* Ptr        ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.type_ptr(o[0], o[1], dbg); }
const Def* Tuple      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.tuple(t, o, dbg); }
const Def* Variadic   ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.variadic(o[0], o[1], dbg); }
const Def* VariantType::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.variant_type(o, dbg); }

/*
 * stub
 */

Def* Axiom   ::stub(const Def* d, World& to, const Def*  , const Def*    ) { assert(d->isa_nominal()); auto axiom = to.lookup(d->name()); assert(axiom); return axiom; }
Def* Lam     ::stub(const Def* d, World& to, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return to.lam(t->as<Pi>(), d->as<Lam>()->cc(), d->as<Lam>()->intrinsic(), dbg); }
Def* Pi      ::stub(const Def* d, World& to, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return to.pi(t, Debug{dbg}); }
Def* Sigma   ::stub(const Def* d, World& to, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return to.sigma(t, d->num_ops(), dbg); }
Def* Universe::stub(const Def*  , World& to, const Def*  , const Def*    ) { return const_cast<Universe*>(to.universe()); }

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Def* type) {
   return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& App        ::stream(std::ostream& os) const { return streamf(os, "{} {}", callee(), arg()); }
std::ostream& Axiom      ::stream(std::ostream& os) const { return streamf(os, "{}", name()); }
std::ostream& Mem        ::stream(std::ostream& os) const { return streamf(os, "mem"); }
std::ostream& Nat        ::stream(std::ostream& os) const { return streamf(os, "nat"); }
std::ostream& Universe   ::stream(std::ostream& os) const { return streamf(os, "□"); }
std::ostream& Variadic   ::stream(std::ostream& os) const { return streamf(os, "«{}; {}»", arity(), body()); }
std::ostream& VariantType::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }
std::ostream& KindArity  ::stream(std::ostream& os) const { return os << "*A"; }
std::ostream& KindMulti  ::stream(std::ostream& os) const { return os << "*M"; }
std::ostream& KindStar   ::stream(std::ostream& os) const { return os << "*"; }

std::ostream& Analyze::stream(std::ostream& os) const {
    stream_list(os << "analyze(", ops().skip_front(), [&](auto def) { os << def; });
    return streamf(os, "; {})", index());
}

std::ostream& Lit::stream(std::ostream& os) const {
    //if (name()) return os << name();
    if (type()->isa<KindArity>()) return streamf(os, "{}ₐ", get());

    if (is_arity(type())) {
        if (type()->isa<Top>()) return streamf(os, "{}T", get());

        std::string s;
        // append utf-8 subscripts in reverse order
        for (size_t aa = as_lit<nat_t>(type()); aa > 0; aa /= 10)
            ((s += char(char(0x80) + char(aa % 10))) += char(0x82)) += char(0xe2);
        std::reverse(s.begin(), s.end());

        return streamf(os, "{}{}", get(), s);
    }

    os << type() << ' ';
#if 0
    if (auto real = type()->isa<Real>()) {
        switch (real->lit_num_bits()) {
            case 16: return os << get<f16>();
            case 32: return os << get<f32>();
            case 64: return os << get<f64>();
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto i = type()->isa<Sint>()) {
        switch (i->lit_num_bits()) {
            case  8: return os << (int) get<s8>();
            case 16: return os << get<s16>();
            case 32: return os << get<s32>();
            case 64: return os << get<s64>();
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto u = type()->isa<Uint>()) {
        switch (u->lit_num_bits()) {
            case  8: return os << (unsigned) get<u8>();
            case 16: return os << get<u16>();
            case 32: return os << get<u32>();
            case 64: return os << get<u64>();
            default: THORIN_UNREACHABLE;
        }
    }
#endif

    return os << get();
}

std::ostream& Bot::stream(std::ostream& os) const {
    if (type()->is_kind()) return streamf(os, "⊥{}", type());
    return streamf(os, "{{⊥: {}}}", type());
}

std::ostream& Top::stream(std::ostream& os) const {
    if (type()->is_kind()) return streamf(os, "⊤{}", type());
    return streamf(os, "{{⊤: {}}}", type());
}

#if 0
std::ostream& Lam::stream(std::ostream& os) const {
    if (isa_nominal())
    return streamf(os, "[{}].{}", name(), body());
}
#endif

std::ostream& Sigma::stream(std::ostream& os) const {
    if (isa_nominal()) return os << unique_name();
    return stream_list(os, ops(), [&](const Def* type) { os << type; }, "[", "]");
}

std::ostream& Tuple::stream(std::ostream& os) const {
#if 0
    // special case for string
    if (std::all_of(ops().begin(), ops().end(), [&](const Def* op) { return op->isa<Lit>(); })) {
        if (auto variadic = type()->isa<Variadic>()) {
            if (auto i = variadic->body()->isa<Sint>()) {
                if (i->lit_num_bits() == 8) {
                    for (auto op : ops()) os << as_lit<char>(op);
                    return os;
                }
            }
        }
    }
#endif

    return stream_list(os, ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& Pack::stream(std::ostream& os) const {
#if 0
    // special case for string
    if (auto variadic = type()->isa<Variadic>()) {
        if (auto i = variadic->body()->isa<Sint>()) {
            if (i->lit_num_bits() == 8) {
                if (auto a = isa_lit<u64>(arity())) {
                    if (auto lit = body()->isa<Lit>()) {
                        for (size_t i = 0, e = *a; i != e; ++i) os << lit->get<char>();
                        return os;
                    }
                }
            }
        }
    }
#endif

    return streamf(os, "‹{}; {}›", arity(), body());
}

std::ostream& Pi::stream(std::ostream& os) const {
    if (is_cn()) {
        if (isa_nominal())
            return streamf(os, "cn {}:{}", param(), domain());
        else
            return streamf(os, "cn {}", domain());
    } else {
        if (isa_nominal())
            return streamf(os, "Π{}:{} -> {}", param(), domain(), codomain());
        else
            return streamf(os, "Π{} -> {}", domain(), codomain());
    }
}

std::ostream& Ptr::stream(std::ostream& os) const {
    os << pointee() << '*';
    switch (auto as = lit_addr_space()) {
        case AddrSpace::Global:   return streamf(os, "[Global]");
        case AddrSpace::Texture:  return streamf(os, "[Tex]");
        case AddrSpace::Shared:   return streamf(os, "[Shared]");
        case AddrSpace::Constant: return streamf(os, "[Constant]");
        default:                  return streamf(os, "[{}]", as);
    }
}

std::ostream& Def::stream(std::ostream& out) const { return out << unique_name(); }

std::ostream& Def::stream_assignment(std::ostream& os) const {
    return streamf(os, "{}: {} = {} {}", unique_name(), type(), op_name(), stream_list(ops(), [&] (const Def* def) { os << def; })) << endl;
}
std::ostream& Lam::stream_head(std::ostream& os) const {
    if (type()->is_cn())
        streamf(os, "cn {} {}: {} @({})", unique_name(), param(), param()->type(), filter());
    else
        streamf(os, "fn {} {}: {} -> {} @({})", unique_name(), param(), param()->type(), codomain(), filter());
    if (is_external()) os << " extern";
    if (cc() == CC::Device) os << " device";
    return os;
}

std::ostream& Lam::stream_body(std::ostream& os) const {
    return streamf(os, "{}", body());
}

void Lam::dump_head() const { stream_head(std::cout) << endl; }
void Lam::dump_body() const { stream_body(std::cout) << endl; }

}
