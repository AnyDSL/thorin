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

const Def* Def::out(size_t i, Dbg dbg) const { return world().extract(this, i, dbg); }
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
        contains_lam_ |= o->contains_lam();
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
    contains_lam_ |= def->contains_lam();
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
    if (tag()                 == Node_Universe) return Sort::Universe;
    if (type()->tag()         == Node_Universe) return Sort::Kind;
    if (type()->type()->tag() == Node_Universe) return Sort::Type;
    assert(type()->type()->type()->tag() == Node_Universe);
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

bool Lam::is_empty() const { return is_bot(body()); }

void Lam::destroy() {
    set_filter(world().lit_false());
    set_body  (world().bot(type()->codomain()));
}

const Param* Lam::param(Dbg dbg) const { return world().param(this->as_nominal<Lam>(), dbg); }
const Def* Lam::param(size_t i, Dbg dbg) const { return world().extract(param(), i, dbg); }
Array<const Def*> Lam::params() const { return Array<const Def*>(num_params(), [&](auto i) { return param(i); }); }

const Def* Lam::mem_param() const {
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (is_mem(p))
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

        for (auto op : def->ops()) {
            if (op->contains_lam())
                enqueue(op);
        }
    }

    return succs;
}

bool Lam::is_intrinsic() const { return intrinsic() != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic() && intrinsic() < Intrinsic::_Accelerator_End; }
void Lam::set_intrinsic() {
    // TODO this is slow and inelegant - but we want to remove this code anyway
    auto n = name();
    if      (n == "cuda")                 extra<Extra>().intrinsic_ = Intrinsic::CUDA;
    else if (n == "nvvm")                 extra<Extra>().intrinsic_ = Intrinsic::NVVM;
    else if (n == "opencl")               extra<Extra>().intrinsic_ = Intrinsic::OpenCL;
    else if (n == "amdgpu")               extra<Extra>().intrinsic_ = Intrinsic::AMDGPU;
    else if (n == "hls")                  extra<Extra>().intrinsic_ = Intrinsic::HLS;
    else if (n == "parallel")             extra<Extra>().intrinsic_ = Intrinsic::Parallel;
    else if (n == "spawn")                extra<Extra>().intrinsic_ = Intrinsic::Spawn;
    else if (n == "sync")                 extra<Extra>().intrinsic_ = Intrinsic::Sync;
    else if (n == "anydsl_create_graph")  extra<Extra>().intrinsic_ = Intrinsic::CreateGraph;
    else if (n == "anydsl_create_task")   extra<Extra>().intrinsic_ = Intrinsic::CreateTask;
    else if (n == "anydsl_create_edge")   extra<Extra>().intrinsic_ = Intrinsic::CreateEdge;
    else if (n == "anydsl_execute_graph") extra<Extra>().intrinsic_ = Intrinsic::ExecuteGraph;
    else if (n == "vectorize")            extra<Extra>().intrinsic_ = Intrinsic::Vectorize;
    else if (n == "pe_info")              extra<Extra>().intrinsic_ = Intrinsic::PeInfo;
    else if (n == "reserve_shared")       extra<Extra>().intrinsic_ = Intrinsic::Reserve;
    else if (n == "atomic")               extra<Extra>().intrinsic_ = Intrinsic::Atomic;
    else if (n == "cmpxchg")              extra<Extra>().intrinsic_ = Intrinsic::CmpXchg;
    else if (n == "undef")                extra<Extra>().intrinsic_ = Intrinsic::Undef;
    else ELOG("unsupported thorin intrinsic");
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }
void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Dbg dbg) { return set_body(world().branch(cond, t, f, mem, dbg)); }
void Lam::app(const Def* callee, Defs args, Dbg dbg) { app(callee, world().tuple(args), dbg); }

void Lam::app(const Def* callee, const Def* arg, Dbg dbg) {
    assert(isa_nominal());
    set_body(world().app(callee, arg, dbg));
}

void Lam::match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Dbg dbg) {
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

bool Pi::is_cn() const { return is_bot(codomain()); }

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

Def::Def(NodeTag tag, RebuildFn rebuild, const Def* type, Defs ops, const Def* dbg)
    : type_(type)
    , rebuild_(rebuild)
    , tag_((unsigned)tag)
    , nominal_(false)
    , contains_lam_(tag == Node_Lam)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(ops.size())
    , debug_(dbg)
{
    std::copy(ops.begin(), ops.end(), ops_ptr());
    hash_ = hash_combine(hash_begin((uint16_t) tag), type->gid());
    for (auto op : ops)
        hash_ = hash_combine(hash_, op->gid());
}

Def::Def(NodeTag tag, StubFn stub, const Def* type, size_t num_ops, const Def* dbg)
    : type_(type)
    , stub_(stub)
    , tag_(tag)
    , nominal_(true)
    , contains_lam_(tag == Node_Lam)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(num_ops)
    , debug_(dbg)
    , hash_(murmur3(gid()))
{
    std::fill_n(ops_ptr(), num_ops, nullptr);
}

App::App(const Def* type, const Def* callee, const Def* arg, const Def* dbg)
    : Def(Node_App, rebuild, type, {callee, arg}, dbg)
{
    //if (is_bot(type)) hash_ = murmur3(gid());
}

Kind::Kind(World& world, NodeTag tag)
    : Def(tag, rebuild, world.universe(), Defs{}, {})
{}

PrimType::PrimType(World& world, PrimTypeTag tag)
    : Def((NodeTag) tag, rebuild, world.kind_star(), Defs{}, {})
{}

MemType::MemType(World& world)
    : Def(Node_MemType, rebuild, world.kind_star(), Defs{}, {})
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

    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->op(i) == other->op(i);
    return result;
}

bool Analyze::equal(const Def* other) const { return Def::equal(other) && this->index()      == other->as<Analyze>()->index(); }
bool Lit    ::equal(const Def* other) const { return Def::equal(other) && this->box()        == other->as<Lit>()->box(); }
bool PtrType::equal(const Def* other) const { return Def::equal(other) && this->addr_space() == other->as<PtrType>()->addr_space(); }

/*
 * rebuild
 */

const Def* Lam        ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { assert(!d->isa_nominal()); return w.lam(t->as<Pi>(), o[0], o[1], dbg); }
const Def* Sigma      ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { assert(!d->isa_nominal()); return w.sigma(t, o, dbg); }
const Def* Analyze    ::rebuild(const Def* d, World& w, const Def* t, Defs o, const Def* dbg) { return w.analyze(t, o, d->as<Analyze>()->index(), dbg); }
const Def* App        ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.app(o[0], o[1], dbg); }
const Def* BotTop     ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.bot_top(is_top(d), t, dbg); }
const Def* Extract    ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.extract(o[0], o[1], dbg); }
const Def* Insert     ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.insert(o[0], o[1], o[2], dbg); }
const Def* Kind       ::rebuild(const Def* d, World& w, const Def*  , Defs  , const Def*    ) { return w.kind(d->as<Kind>()->tag()); }
const Def* Lit        ::rebuild(const Def* d, World& w, const Def* t, Defs  , const Def* dbg) { return w.lit(t, d->as<Lit>()->box(), dbg); }
const Def* MemType    ::rebuild(const Def*  , World& w, const Def*  , Defs  , const Def*    ) { return w.mem_type(); }
const Def* Pack       ::rebuild(const Def*  , World& w, const Def* t, Defs o, const Def* dbg) { return w.pack(t->arity(), o[0], dbg); }
const Def* Param      ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.param(o[0]->as_nominal<Lam>(), dbg); }
const Def* Pi         ::rebuild(const Def*  , World& w, const Def*  , Defs o, const Def* dbg) { return w.pi(o[0], o[1], dbg); }
const Def* PrimType   ::rebuild(const Def* d, World& w, const Def*  , Defs  , const Def*    ) { return w.type(d->as<PrimType>()->primtype_tag()); }
const Def* PtrType    ::rebuild(const Def* d, World& w, const Def*  , Defs o, const Def*    ) { return w.ptr_type(o[0], d->as<PtrType>()->addr_space()); }
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
std::ostream& MemType    ::stream(std::ostream& os) const { return streamf(os, "mem"); }
std::ostream& Universe   ::stream(std::ostream& os) const { return streamf(os, "□"); }
std::ostream& Variadic   ::stream(std::ostream& os) const { return streamf(os, "«{}; {}»", arity(), body()); }
std::ostream& VariantType::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }

std::ostream& Kind::stream(std::ostream& os) const {
    switch (tag()) {
        case Node_KindArity: return os << "*A";
        case Node_KindMulti: return os << "*M";
        case Node_KindStar : return os << "*";
        default: THORIN_UNREACHABLE;
    }
}

std::ostream& Analyze::stream(std::ostream& os) const {
    stream_list(os << "analyze(", ops(), [&](auto def) { os << def; });
    return streamf(os, "; {})", index());
}

std::ostream& Lit::stream(std::ostream& os) const {
    //if (name()) return os << name();
    if (is_kind_arity(type())) return streamf(os, "{}ₐ", box().get<u64>());

    if (is_arity(type())) {
        if (is_top(type())) return streamf(os, "{}T", box().get<u64>());

        std::string s;
        // append utf-8 subscripts in reverse order
        for (size_t aa = as_lit<u64>(type()); aa > 0; aa /= 10)
            ((s += char(char(0x80) + char(aa % 10))) += char(0x82)) += char(0xe2);
        std::reverse(s.begin(), s.end());

        return streamf(os, "{}{}", box().get<u64>(), s);
    }

    // special case for char
    if (auto prim_type = type()->isa<PrimType>()) {
        if (prim_type->primtype_tag() == PrimTypeTag::PrimType_qs8) {
            char c = box().get<qs8>();
            if (0x21 <= c && c <= 0x7e) return os << 'c'; // printable char range
        }
    }

    os << type() << ' ';
    if (auto prim_type = type()->isa<PrimType>()) {
        auto tag = prim_type->primtype_tag();

        // print i8 as ints
        switch (tag) {
            case PrimType_qs8: return os << (int) box().get_qs8();
            case PrimType_ps8: return os << (int) box().get_ps8();
            case PrimType_qu8: return os << (unsigned) box().get_qu8();
            case PrimType_pu8: return os << (unsigned) box().get_pu8();
            default:
                switch (tag) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return os << box().get_##M();
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
        }
    } else {
        return os << box().get_u64();
    }
}

std::ostream& BotTop::stream(std::ostream& os) const {
    auto op = is_bot(this) ? "⊥" : "⊤";
    if (type()->is_kind())
        return streamf(os, "{}{}", op, type());
    return streamf(os, "{{{}: {}}}", op, type());
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
                    for (auto op : ops()) os << as_lit<qs8>(op);
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
                        for (size_t i = 0, e = *a; i != e; ++i) os << lit->box().get<qs8>();
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
    switch (addr_space()) {
        case AddrSpace::Global:   os << "[Global]";   break;
        case AddrSpace::Texture:  os << "[Tex]";      break;
        case AddrSpace::Shared:   os << "[Shared]";   break;
        case AddrSpace::Constant: os << "[Constant]"; break;
        default: /* ignore unknown address space */   break;
    }
    return os;
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
