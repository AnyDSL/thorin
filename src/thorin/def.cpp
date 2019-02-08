#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * helpers
 */

bool is_unit(const Def* def) {
    return def->type() == def->world().sigma();
}

bool is_const(const Def* def) {
    unique_stack<DefSet> stack;
    stack.push(def);

    while (!stack.empty()) {
        auto def = stack.pop();
        if (def->isa<Param>()) return false;
        if (def->isa<Hlt>()) return false;
        if (!def->isa_nominal()) {
            for (auto op : def->ops())
                stack.push(op);
        }
        // lams are always const
    }

    return true;
}

bool is_primlit(const Def* def, int64_t val) {
    if (auto lit = def->isa<Lit>()) {
        if (auto prim_type = lit->type()->isa<PrimType>()) {
            switch (prim_type->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->box().get_##T() == T(val);
#include "thorin/tables/primtypetable.h"
                case PrimType_bool: return lit->box().get_bool() == bool(val);
                default: ; // FALLTHROUGH
            }
        }
    }

    return false;
}

bool is_minus_zero(const Def* def) {
    if (auto lit = def->isa<Lit>()) {
        if (auto prim_type = lit->type()->isa<PrimType>()) {
            switch (prim_type->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->box().get_##M() == M(0);
#define THORIN_F_TYPE(T, M) case PrimType_##T: return lit->box().get_##M() == M(-0.0);
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        }
    }
    return false;
}

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (is_top(app->arg(i)))
            nargs.push_back(src_app->arg(i));
    }

    src->app(dst, nargs, src_app->debug());
}

bool is_param(const Def* def) {
    if (def->isa<Param>()) return true;
    if (auto extract = def->isa<Extract>()) return extract->agg()->isa<Param>();
    return false;
}

Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->agg()->as<Param>()->lam();
    return def->as<Param>()->lam();
}

// TODO
size_t get_param_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return as_lit<size_t>(extract->index());
    assert(def->isa<Param>());
    return 0;
}

std::vector<Peek> peek(const Def* param) {
    std::vector<Peek> peeks;
    size_t index = get_param_index(param);
    for (auto use : get_param_lam(param)->uses()) {
        if (auto app = use->isa<App>()) {
            for (auto use : app->uses()) {
                if (auto pred = use->isa_nominal<Lam>()) {
                    if (pred->body() == app)
                        peeks.emplace_back(app->arg(index), pred);
                }
            }
        }
    }

    return peeks;
}

bool visit_uses(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    if (!lam->is_intrinsic()) {
        for (auto use : lam->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto lam = def->isa_nominal<Lam>())
                if (func(lam))
                    return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    return visit_uses(lam, [&] (Lam* lam) {
        if (auto callee = lam->app()->callee()->isa_nominal<Lam>())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
}

bool is_tuple_arg_of_app(const Def* def) {
    if (!def->isa<Tuple>()) return false;
    for (auto& use : def->uses()) {
        if (use.index() == 1 && use->isa<App>())
            continue;
        if (!is_tuple_arg_of_app(use.def()))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------

/*
 * Def
 */

Debug Def::debug_history() const {
#if THORIN_ENABLE_CHECKS
    return world().track_history() ? Debug(loc(), unique_name()) : debug();
#else
    return debug();
#endif
}

void Def::finalize() {
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        auto o = op(i);
        assert(o != nullptr);
        contains_lam_ |= o->contains_lam();
        order_ = std::max(order_, o->order_);
        const auto& p = o->uses_.emplace(i, this);
        assert_unused(p.second);
    }

    if (isa<Pi>()) ++order_;
}

void Def::set(size_t i, const Def* def) {
    assert(def && "setting null pointer");

    if (op(i) != nullptr) unset(i);

    assert(i < num_ops() && "index out of bounds");
    ops_ptr()[i] = def;
    contains_lam_ |= def->contains_lam();
    order_ = std::max(order_, def->order_);
    const auto& p = def->uses_.emplace(i, this);
    assert_unused(p.second);
}

void Def::unset(size_t i) {
    assert(i < num_ops() && "index out of bounds");
    auto def = op(i);
    assert(def->uses_.contains(Use(i, this)));
    def->uses_.erase(Use(i, this));
    assert(!def->uses_.contains(Use(i, this)));
    ops_ptr()[i] = nullptr;
}

std::string Def::unique_name() const {
    std::ostringstream oss;
    oss << name() << '_' << gid();
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

size_t App::num_args() const {
    if (auto sigma = arg()->type()->isa<Sigma>())
        return sigma->num_ops();
    return 1;
}

const Def* App::arg(size_t i) const { return callee()->world().extract(arg(), i); }

Array<const Def*> App::args() const {
    size_t n = num_args();
    Array<const Def*> args(n);
    for (size_t i = 0; i != n; ++i)
        args[i] = arg(i);
    return args;
}

/*
 * Lam
 */

void Lam::destroy() {
    set_filter(world().tuple(Array<const Def*>(type()->num_domains(), world().lit(false))));
    set_body  (world().bot(type()->codomain()));
}

const Param* Lam::param(Debug dbg) const { return world().param(this->as_nominal<Lam>(), dbg); }
void Lam::set_filter(Defs filter) { set_filter(world().tuple(filter)); }
const Def* Lam::param(size_t i, Debug dbg) const { return world().extract(param(), i, dbg); }

Array<const Def*> Lam::params() const {
    size_t n = num_params();
    Array<const Def*> params(n);
    for (size_t i = 0; i != n; ++i)
        params[i] = param(i);
    return params;
}

size_t Lam::num_params() const {
    if (auto sigma = param()->type()->isa<Sigma>())
        return sigma->num_ops();
    return 1;
}

const Def* Lam::filter(size_t i) const { return world().extract(filter(), i); }

Array<const Def*> Lam::filters() const {
    size_t n = num_filters();
    Array<const Def*> filters(n);
    for (size_t i = 0; i != n; ++i)
        filters[i] = filter(i);
    return filters;
}

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

void Lam::make_external() { return world().add_external(this); }
void Lam::make_internal() { return world().remove_external(this); }
bool Lam::is_external() const { return world().is_external(this); }
bool Lam::is_intrinsic() const { return intrinsic() != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic() && intrinsic() < Intrinsic::_Accelerator_End; }
void Lam::set_intrinsic() {
    if      (name() == "cuda")                 extra<Extra>().intrinsic_ = Intrinsic::CUDA;
    else if (name() == "nvvm")                 extra<Extra>().intrinsic_ = Intrinsic::NVVM;
    else if (name() == "opencl")               extra<Extra>().intrinsic_ = Intrinsic::OpenCL;
    else if (name() == "amdgpu")               extra<Extra>().intrinsic_ = Intrinsic::AMDGPU;
    else if (name() == "hls")                  extra<Extra>().intrinsic_ = Intrinsic::HLS;
    else if (name() == "parallel")             extra<Extra>().intrinsic_ = Intrinsic::Parallel;
    else if (name() == "spawn")                extra<Extra>().intrinsic_ = Intrinsic::Spawn;
    else if (name() == "sync")                 extra<Extra>().intrinsic_ = Intrinsic::Sync;
    else if (name() == "anydsl_create_graph")  extra<Extra>().intrinsic_ = Intrinsic::CreateGraph;
    else if (name() == "anydsl_create_task")   extra<Extra>().intrinsic_ = Intrinsic::CreateTask;
    else if (name() == "anydsl_create_edge")   extra<Extra>().intrinsic_ = Intrinsic::CreateEdge;
    else if (name() == "anydsl_execute_graph") extra<Extra>().intrinsic_ = Intrinsic::ExecuteGraph;
    else if (name() == "vectorize")            extra<Extra>().intrinsic_ = Intrinsic::Vectorize;
    else if (name() == "pe_info")              extra<Extra>().intrinsic_ = Intrinsic::PeInfo;
    else if (name() == "reserve_shared")       extra<Extra>().intrinsic_ = Intrinsic::Reserve;
    else if (name() == "atomic")               extra<Extra>().intrinsic_ = Intrinsic::Atomic;
    else if (name() == "cmpxchg")              extra<Extra>().intrinsic_ = Intrinsic::CmpXchg;
    else if (name() == "undef")                extra<Extra>().intrinsic_ = Intrinsic::Undef;
    else ELOG("unsupported thorin intrinsic");
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }
void Lam::branch(const Def* cond, const Def* t, const Def* f, Debug dbg) { return app(world().branch(), {cond, t, f}, dbg); }
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

Def::Def(NodeTag tag, const Def* type, Defs ops, Debug dbg)
    : type_(type)
    , tag_((unsigned)tag)
    , is_value_(is_term())
    , nominal_(false)
    , dependent_(false)
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

Def::Def(NodeTag tag, const Def* type, size_t num_ops, Debug dbg)
    : type_(type)
    , tag_(tag)
    , is_value_(is_term())
    , nominal_(true)
    , dependent_(false)
    , contains_lam_(tag == Node_Lam)
    , order_(0)
    , gid_(world().next_gid())
    , num_ops_(num_ops)
    , debug_(dbg)
    , hash_(murmur3(gid()))
{
    std::fill_n(ops_ptr(), num_ops, nullptr);
}

static inline const char* kind2str(NodeTag tag) {
    switch (tag) {
        case Node_KindArity: return "*A";
        case Node_KindMulti: return "*M";
        case Node_KindStar:  return "*";
        default: THORIN_UNREACHABLE;
    }
}

Kind::Kind(World& world, NodeTag tag)
    : Def(tag, world.universe(), Defs{}, {kind2str(tag)})
{}

PrimType::PrimType(World& world, PrimTypeTag tag, Debug dbg)
    : Def((NodeTag) tag, world.kind_star(), Defs{}, dbg)
{}

MemType::MemType(World& world)
    : Def(Node_MemType, world.kind_star(), Defs{}, {"mem"})
{}

FrameType::FrameType(World& world)
    : Def(Node_FrameType, world.kind_star(), Defs{}, {"frame"})
{}

/*
 * arity
 */

const Def* Def  ::arity() const { return is_value() ? type()->arity() : world().lit_arity_1(); }
const Def* Sigma::arity() const { return world().lit_arity(num_ops()); }

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

bool Lit    ::equal(const Def* other) const { return Def::equal(other) && this->box()        == other->as<Lit>()->box(); }
bool PtrType::equal(const Def* other) const { return Def::equal(other) && this->addr_space() == other->as<PtrType>()->addr_space(); }
bool Var    ::equal(const Def* other) const { return Def::equal(other) && this->index()      == other->as<Var>()->index(); }

/*
 * rebuild
 */

const Def* Universe   ::rebuild(World&   , const Def*  , Defs    ) const { THORIN_UNREACHABLE; }
const Def* Lam        ::rebuild(World& to, const Def* t, Defs ops) const { assert(!isa_nominal()); return to.lam(t->as<Pi>(), ops[0], ops[1], debug()); }
const Def* Sigma      ::rebuild(World& to, const Def* t, Defs ops) const { assert(!isa_nominal()); return to.sigma(t, ops, debug()); }
const Def* App        ::rebuild(World& to, const Def*  , Defs ops) const { return to.app(ops[0], ops[1], debug()); }
const Def* BotTop     ::rebuild(World& to, const Def* t, Defs    ) const { return to.bot_top(is_top(this), t, debug()); }
const Def* Extract    ::rebuild(World& to, const Def*  , Defs ops) const { return to.extract(ops[0], ops[1], debug()); }
const Def* FrameType  ::rebuild(World& to, const Def*  , Defs    ) const { return to.frame_type(); }
const Def* Insert     ::rebuild(World& to, const Def*  , Defs ops) const { return to.insert(ops[0], ops[1], ops[2], debug()); }
const Def* Kind       ::rebuild(World& to, const Def*  , Defs    ) const { return to.kind(tag()); }
const Def* MemType    ::rebuild(World& to, const Def*  , Defs    ) const { return to.mem_type(); }
const Def* Pack       ::rebuild(World& to, const Def* t, Defs ops) const { return to.pack(t->arity(), ops[0], debug()); }
const Def* Param      ::rebuild(World& to, const Def*  , Defs ops) const { return to.param(ops[0]->as_nominal<Lam>(), debug()); }
const Def* Pi         ::rebuild(World& to, const Def*  , Defs ops) const { return to.pi(ops[0], ops[1], debug()); }
const Def* PrimType   ::rebuild(World& to, const Def*  , Defs    ) const { return to.type(primtype_tag()); }
const Def* PtrType    ::rebuild(World& to, const Def*  , Defs ops) const { return to.ptr_type(ops[0], addr_space()); }
const Def* Tuple      ::rebuild(World& to, const Def* t, Defs ops) const { return to.tuple(t, ops, debug()); }
const Def* Var        ::rebuild(World& to, const Def* t, Defs    ) const { return to.var(t, index(), debug()); }
const Def* Variadic   ::rebuild(World& to, const Def*  , Defs ops) const { return to.variadic(ops[0], ops[1], debug()); }
const Def* VariantType::rebuild(World& to, const Def*  , Defs ops) const { return to.variant_type(ops, debug()); }

/*
 * stub
 */

Lam*   Lam  ::stub(World& to, const Def* type) const { assert(isa_nominal()); return to.lam(type->as<Pi>(), cc(), intrinsic(), debug()); }
Sigma* Sigma::stub(World& to, const Def* type) const { assert(isa_nominal()); return to.sigma(type, num_ops(), debug()); }
Universe* Universe::stub(World& to, const Def*) const { return const_cast<Universe*>(to.universe()); }

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Def* type) {
   return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& App        ::stream(std::ostream& os) const { return streamf(os, "{} {}", callee(), arg()); }
std::ostream& FrameType  ::stream(std::ostream& os) const { return streamf(os, "frame"); }
std::ostream& Kind       ::stream(std::ostream& os) const { return streamf(os, "{}", name()); }
std::ostream& MemType    ::stream(std::ostream& os) const { return streamf(os, "mem"); }
std::ostream& Pack       ::stream(std::ostream& os) const { return streamf(os, "‹{}; {}›", arity(), body()); }
std::ostream& Universe   ::stream(std::ostream& os) const { return streamf(os, "□"); }
std::ostream& Var        ::stream(std::ostream& os) const { return streamf(os, "<{}:{}>", index(), type()); }
std::ostream& Variadic   ::stream(std::ostream& os) const { return streamf(os, "«{}; {}»", arity(), body()); }
std::ostream& VariantType::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }

std::ostream& Lit::stream(std::ostream& os) const {
    if (!name().empty()) return os << name();

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
    return stream_list(os, ops(), [&](const Def* type) { os << type; }, "(", ")");
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
