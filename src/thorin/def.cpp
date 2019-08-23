#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/primop.h"
#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

/*
 * Def
 */

const Def* Def::out(size_t i, Debug dbg) const { return world().extract(this, i, dbg); }
size_t Def::num_outs() const { return as_lit<u64>(arity()); }

// TODO
const Def* Def::debug_history() const {
//#if THORIN_ENABLE_CHECKS
    //return world().track_history() ? Debug(loc(), world().tuple_str(unique_name())) : debug();
//#else
    return debug();
//#endif
}

std::string Def::name() const     { return debug() ?   tuple2str(debug()->out(0)) : std::string{}; }
std::string Def::filename() const { return debug() ?   tuple2str(debug()->out(1)) : std::string{}; }
u64 Def::front_line() const  { return debug() ? as_lit<u64>(debug()->out(2)) : std::numeric_limits<u64>::max(); }
u64 Def::front_col() const   { return debug() ? as_lit<u64>(debug()->out(3)) : std::numeric_limits<u64>::max(); }
u64 Def::back_line() const   { return debug() ? as_lit<u64>(debug()->out(4)) : std::numeric_limits<u64>::max(); }
u64 Def::back_col() const    { return debug() ? as_lit<u64>(debug()->out(5)) : std::numeric_limits<u64>::max(); }

std::string Def::loc() const {
    std::ostringstream os;
    os << filename() << ':';

    if (front_col() == u64(-1) || back_col() == u64(-1)) {
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
    assert(def && "setting null pointer");

    if (op(i) != nullptr) unset(i);

    assert(i < num_ops() && "index out of bounds");
    ops_ptr()[i] = def;
    order_ = std::max(order_, def->order_);
    const auto& p = def->uses_.emplace(this, i);
    assert_unused(p.second);
    return this;
}

void Def::make_external() { return world().make_external(this); }
void Def::make_internal() { return world().make_internal(this); }
bool Def::is_external() const { return world().is_external(this); }

void Def::unset(size_t i) {
    assert(i < num_ops() && "index out of bounds");
    auto def = op(i);
    assert(def->uses_.contains(Use(this, i)));
    def->uses_.erase(Use(this, i));
    assert(!def->uses_.contains(Use(this, i)));
    ops_ptr()[i] = nullptr;
}

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
    if (tag()                 == Tag::Universe) return Sort::Universe;
    if (type()->tag()         == Tag::Universe) return Sort::Kind;
    if (type()->type()->tag() == Tag::Universe) return Sort::Type;
    assert(type()->type()->type()->tag() == Tag::Universe);
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
 * App
 */

const Def* App::arg(size_t i) const { return callee()->world().extract(arg(), i); }
Array<const Def*> App::args() const { return Array<const Def*>(num_args(), [&](auto i) { return arg(i); }); }

/*
 * Lam
 */

bool Lam::is_empty() const { return body()->isa<Bot>(); }

void Lam::destroy() {
    set_filter(world().lit_false());
    set_body  (world().bot(type()->codomain()));
}

const Param* Lam::param(Debug dbg) const { return world().param(this->as_nominal<Lam>(), dbg); }
const Def* Lam::param(size_t i, Debug dbg) const { return world().extract(param(), i, dbg); }
Array<const Def*> Lam::params() const { return Array<const Def*>(num_params(), [&](auto i) { return param(i); }); }

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
void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Debug dbg) { return set_body(world().branch(cond, t, f, mem, dbg)); }
void Lam::app(const Def* callee, Defs args, Debug dbg) { app(callee, world().tuple(args), dbg); }

void Lam::app(const Def* callee, const Def* arg, Debug dbg) {
    assert(isa_nominal());
    set_body(world().app(callee, arg, dbg));
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

bool Pi::is_cn() const { return codomain()->isa<Bot>(); }

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

//------------------------------------------------------------------------------

/*
 * constructors
 */

Def::Def(uint16_t tag, RebuildFn rebuild, const Def* type, Defs ops, uint64_t flags, const Def* dbg)
    : type_(type)
    , rebuild_(rebuild)
    , debug_(dbg)
    , flags_(flags)
    , tag_((unsigned)tag)
    , nominal_(false)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(ops.size())
{
    std::copy(ops.begin(), ops.end(), ops_ptr());
    hash_ = hash_combine(hash_begin((uint16_t) tag), type->gid(), flags_);
    for (auto op : ops)
        hash_ = hash_combine(hash_, op->gid());
}

Def::Def(uint16_t tag, StubFn stub, const Def* type, size_t num_ops, uint64_t flags, const Def* dbg)
    : type_(type)
    , stub_(stub)
    , debug_(dbg)
    , flags_(flags)
    , tag_(tag)
    , nominal_(true)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(num_ops)
    , hash_(murmur3(gid()))
{
    std::fill_n(ops_ptr(), num_ops, nullptr);
}

App::App(const Def* type, const Def* callee, const Def* arg, const Def* dbg)
    : Def(Tag, rebuild, type, {callee, arg}, 0, dbg)
{}

KindArity::KindArity(World& world)
    : Def(Tag, rebuild, world.universe(), Defs{}, 0, nullptr)
{}

KindMulti::KindMulti(World& world)
    : Def(Tag, rebuild, world.universe(), Defs{}, 0, nullptr)
{}

KindStar::KindStar(World& world)
    : Def(Tag, rebuild, world.universe(), Defs{}, 0, nullptr)
{}

Nat::Nat(World& world)
    : Def(Tag, rebuild, world.kind_star(), Defs{}, 0, nullptr)
{}

PrimType::PrimType(World& world, PrimTypeTag tag)
    : Def(Tag, rebuild, world.kind_star(), Defs{}, tag, nullptr)
{}

Mem::Mem(World& world)
    : Def(Tag, rebuild, world.kind_star(), Defs{}, 0, nullptr)
{}

/*
 * arity
 */

const Def* Def  ::arity() const { return is_term() ? type()->arity() : world().lit_arity_1(); }
const Def* Sigma::arity() const { return world().lit_arity(num_ops()); }
u64 Def::lit_arity() const { return as_lit<u64>(arity()); }

/*
 * equal
 */

bool Def::equal(const Def* other) const {
    if (this->isa_nominal() || other->isa_nominal())
        return this == other;

    bool result = this->tag() == other->tag() && this->flags() == other->flags() && this->num_ops() == other->num_ops() && this->type() == other->type();
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
const Def* Lit        ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.lit(t, as_lit<u64>(d), dbg); }
const Def* Nat        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_nat(); }
const Def* Mem        ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.type_mem(); }
const Def* Pack       ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.pack(t->arity(), o[0], dbg); }
const Def* Param      ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.param(o[0]->as_nominal<Lam>(), dbg); }
const Def* Pi         ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.pi(o[0], o[1], dbg); }
const Def* PrimType   ::rebuild(const Def* d, World& w, const Def*  , Defs  , const Def*    ) { return w.type(d->as<PrimType>()->primtype_tag()); }
const Def* PtrType    ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.ptr_type(o[0], o[1], dbg); }
const Def* Tuple      ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.tuple(t, o, dbg); }
const Def* Variadic   ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.variadic(o[0], o[1], dbg); }
const Def* VariantType::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.variant_type(o, dbg); }

/*
 * stub
 */

Def* Lam     ::stub(const Def* d, World& to, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return to.lam(t->as<Pi>(), d->as<Lam>()->cc(), d->as<Lam>()->intrinsic(), dbg); }
Def* Sigma   ::stub(const Def* d, World& to, const Def* t, const Def* dbg) { assert(d->isa_nominal()); return to.sigma(t, d->num_ops(), dbg); }
Def* Universe::stub(const Def*  , World& to, const Def*  , const Def*    ) { return const_cast<Universe*>(to.universe()); }

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Def* type) {
   return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& App        ::stream(std::ostream& os) const { return streamf(os, "{} {}", callee(), arg()); }
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
        for (size_t aa = as_lit<u64>(type()); aa > 0; aa /= 10)
            ((s += char(char(0x80) + char(aa % 10))) += char(0x82)) += char(0xe2);
        std::reverse(s.begin(), s.end());

        return streamf(os, "{}{}", get(), s);
    }

    // special case for char
    if (auto prim_type = type()->isa<PrimType>()) {
        if (prim_type->primtype_tag() == PrimTypeTag::PrimType_qs8) {
            char c = get<char>();
            if (0x21 <= c && c <= 0x7e) return os << 'c'; // printable char range
        }
    }

    os << type() << ' ';
    if (auto prim_type = type()->isa<PrimType>()) {
        auto tag = prim_type->primtype_tag();

        // print i8 as ints
        switch (tag) {
            case PrimType_qs8: return os << (int) get<char>();
            case PrimType_ps8: return os << (int) get<char>();
            case PrimType_qu8: return os << (unsigned) get<char>();
            case PrimType_pu8: return os << (unsigned) get<char>();
            default:
                switch (tag) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return os << get<M>();
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
        }
    } else {
        return os << get();
    }
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
    // special case for string
    if (std::all_of(ops().begin(), ops().end(), [&](const Def* op) { return op->isa<Lit>(); })) {
        if (auto variadic = type()->isa<Variadic>()) {
            if (auto prim_type = variadic->body()->isa<PrimType>()) {
                if (prim_type->primtype_tag() == PrimTypeTag::PrimType_qs8) {
                    for (auto op : ops()) os << as_lit<char>(op);
                    return os;
                }
            }
        }
    }

    return stream_list(os, ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& Pack::stream(std::ostream& os) const {
    // special case for string
    if (auto variadic = type()->isa<Variadic>()) {
        if (auto prim_type = variadic->body()->isa<PrimType>()) {
            if (prim_type->primtype_tag() == PrimTypeTag::PrimType_qs8) {
                if (auto a = isa_lit<u64>(arity())) {
                    if (auto lit = body()->isa<Lit>()) {
                        for (size_t i = 0, e = *a; i != e; ++i) os << lit->get<char>();
                        return os;
                    }
                }
            }
        }
    }

    return streamf(os, "‹{}; {}›", arity(), body());
}

std::ostream& Pi::stream(std::ostream& os) const {
    return is_cn()
        ? streamf(os, "cn {}", domain())
        : streamf(os, "Π{} -> {}", domain(), codomain());
}

std::ostream& PtrType::stream(std::ostream& os) const {
    os << pointee() << '*';
    switch (auto as = lit_addr_space()) {
        case AddrSpace::Global:   return streamf(os, "[Global]");
        case AddrSpace::Texture:  return streamf(os, "[Tex]");
        case AddrSpace::Shared:   return streamf(os, "[Shared]");
        case AddrSpace::Constant: return streamf(os, "[Constant]");
        default:                  return streamf(os, "[{}]", as);
    }
}

std::ostream& PrimType::stream(std::ostream& os) const {
    switch (primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: return os << #T;
#include "thorin/tables/primtypetable.h"
          default: THORIN_UNREACHABLE;
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
