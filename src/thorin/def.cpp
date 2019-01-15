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

size_t Def::gid_counter_ = 1;

/*
 * helpers
 */

bool is_unit(const Def* def) {
    return def->type() == def->world().unit();
}

bool is_const(const Def* def) {
    unique_stack<DefSet> stack;
    stack.push(def);

    while (!stack.empty()) {
        auto def = stack.pop();
        if (def->isa<Param>()) return false;
        if (def->isa<Hlt>()) return false;
        if (def->isa<PrimOp>()) {
            for (auto op : def->ops())
                stack.push(op);
        }
        // lams are always const
    }

    return true;
}

size_t vector_length(const Def* def) {
    if (auto vector_type = def->isa<VectorType>())
        return vector_type->length();
    return def->type()->as<VectorType>()->length();
}


bool is_primlit(const Def* def, int64_t val) {
    if (auto lit = def->isa<PrimLit>()) {
        switch (lit->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->value().get_##T() == T(val);
#include "thorin/tables/primtypetable.h"
            case PrimType_bool: return lit->value().get_bool() == bool(val);
            default: ; // FALLTHROUGH
        }
    }

    if (auto vector = def->isa<Vector>()) {
        for (auto op : vector->ops()) {
            if (!is_primlit(op, val))
                return false;
        }
        return true;
    }
    return false;
}

bool is_minus_zero(const Def* def) {
    if (auto lit = def->isa<PrimLit>()) {
        Box box = lit->value();
        switch (lit->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return box.get_##M() == M(0);
#define THORIN_F_TYPE(T, M) case PrimType_##T: return box.get_##M() == M(-0.0);
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }
    }
    return false;
}

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (app->arg(i)->isa<Top>())
            nargs.push_back(src_app->arg(i));
    }

    src->app(dst, nargs, src_app->debug());
}

Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->agg()->as<Param>()->lam();
    return def->as<Param>()->lam();
}

size_t get_param_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return primlit_value<u64>(extract->index()->as<PrimLit>());
    assert(def->isa<Param>());
    return 0;
}

std::vector<Peek> peek(const Def* param) {
    std::vector<Peek> peeks;
    size_t index = get_param_index(param);
    for (auto use : get_param_lam(param)->uses()) {
        if (auto pred = use->isa_lam()) {
            if (auto app = pred->app()) {
                if (use.index() == 1) // is it an arg of pred?
                    peeks.emplace_back(app->arg(index), pred);
            }
        }
    }

    return peeks;
}

bool visit_uses(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    if (!lam->is_intrinsic()) {
        for (auto use : lam->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto lam = def->isa_lam())
                if (func(lam))
                    return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    return visit_uses(lam, [&] (auto lam) {
        if (auto callee = lam->app()->callee()->isa_lam())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
}

bool is_passed_to_accelerator(Lam* lam, bool include_globals) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->is_accelerator(); }, include_globals);
}

bool is_passed_to_intrinsic(Lam* lam, Intrinsic intrinsic, bool include_globals) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->intrinsic() == intrinsic; }, include_globals);
}

const Def* merge_sigma(const Def* a, const Def* b) {
    auto x = a->isa<Sigma>();
    auto y = b->isa<Sigma>();
    auto& w = a->world();

    if ( x &&  y) return w.sigma(concat(x->ops(), y->ops()));
    if ( x && !y) return w.sigma(concat(x->ops(), b       ));
    if (!x &&  y) return w.sigma(concat(a,        y->ops()));

    assert(!x && !y);
    return w.sigma({a, b});
}

//------------------------------------------------------------------------------

/*
 * Def
 */

Debug Def::debug_history() const {
#if THORIN_ENABLE_CHECKS
    return world().track_history() ? Debug(location(), unique_name()) : debug();
#else
    return debug();
#endif
}

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(def && "setting null pointer");
    ops_[i] = def;
    contains_lam_ |= def->contains_lam();
    assert(!def->uses_.contains(Use(i, this)));
    const auto& p = def->uses_.emplace(i, this);
    assert_unused(p.second);
}

void Def::unregister_uses() const {
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        unregister_use(i);
}

void Def::unregister_use(size_t i) const {
    auto def = ops_[i];
    assert(def->uses_.contains(Use(i, this)));
    def->uses_.erase(Use(i, this));
    assert(!def->uses_.contains(Use(i, this)));
}

void Def::unset_op(size_t i) {
    assert(ops_[i] && "must be set");
    unregister_use(i);
    ops_[i] = nullptr;
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
            def->update_op(index, with);
        }

        uses_.clear();
        substitute_ = with;
    }
}

void Def::dump() const {
    auto primop = this->isa<PrimOp>();
    if (primop && primop->num_ops() > 1)
        primop->stream_assignment(std::cout);
    else {
        std::cout << this;
        std::cout << std::endl;
    }
}

Lam* Def::as_lam() const { return const_cast<Lam*>(scast<Lam>(this)); }
Lam* Def::isa_lam() const { return const_cast<Lam*>(dcast<Lam>(this)); }

// TODO remove
const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return world().ptr_type(ptr->pointee());
    return world().type(as<PrimType>()->primtype_tag());
}

/*
 * App
 */

size_t App::num_args() const {
    if (auto sigma = arg()->type()->isa<Sigma>())
        return sigma->num_ops();
    return 1;
}

const Def* App::arg(size_t i) const {
    if (arg()->type()->isa<Sigma>())
        return callee()->world().extract(arg(), i);
    return arg();
}

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

const Param* Lam::param(Debug dbg) const { return world().param(this->as_lam(), dbg); }
bool Lam::is_empty() const { return body()->isa<Top>(); }
void Lam::set_filter(Defs filter) { set_filter(world().tuple(filter)); }

size_t Lam::num_params() const {
    if (auto sigma = param()->type()->isa<Sigma>())
        return sigma->num_ops();
    return 1;
}

const Def* Lam::param(size_t i, Debug dbg) const {
    if (param()->type()->isa<Sigma>())
        return world().extract(param(), i, dbg);
    assert(i == 0);
    return param();
}

Array<const Def*> Lam::params() const {
    size_t n = num_params();
    Array<const Def*> params(n);
    for (size_t i = 0; i != n; ++i)
        params[i] = param(i);
    return params;
}

const Def* Lam::filter(size_t i) const {
    if (filter()->type()->isa<Sigma>())
        return world().extract(filter(), i);
    return filter();
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
        if (p->order() >= 1) {
            assertf(result == nullptr, "only one ret_param allowed");
            result = p;
        }
    }
    return result;
}

void Lam::destroy_filter() {
    update_op(0, world().literal_bool(false));
}

void Lam::destroy_body() {
    update_op(0, world().literal_bool(false));
    update_op(1, world().top(type()->codomain()));
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
        if (auto lam = use->isa_lam()) {
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
        if (auto lam = def->isa_lam()) {
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
bool Lam::is_intrinsic() const { return intrinsic_ != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic_ && intrinsic_ < Intrinsic::_Accelerator_End; }
void Lam::set_intrinsic() {
    if      (name() == "cuda")                 intrinsic_ = Intrinsic::CUDA;
    else if (name() == "nvvm")                 intrinsic_ = Intrinsic::NVVM;
    else if (name() == "opencl")               intrinsic_ = Intrinsic::OpenCL;
    else if (name() == "amdgpu")               intrinsic_ = Intrinsic::AMDGPU;
    else if (name() == "hls")                  intrinsic_ = Intrinsic::HLS;
    else if (name() == "parallel")             intrinsic_ = Intrinsic::Parallel;
    else if (name() == "spawn")                intrinsic_ = Intrinsic::Spawn;
    else if (name() == "sync")                 intrinsic_ = Intrinsic::Sync;
    else if (name() == "anydsl_create_graph")  intrinsic_ = Intrinsic::CreateGraph;
    else if (name() == "anydsl_create_task")   intrinsic_ = Intrinsic::CreateTask;
    else if (name() == "anydsl_create_edge")   intrinsic_ = Intrinsic::CreateEdge;
    else if (name() == "anydsl_execute_graph") intrinsic_ = Intrinsic::ExecuteGraph;
    else if (name() == "vectorize")            intrinsic_ = Intrinsic::Vectorize;
    else if (name() == "pe_info")              intrinsic_ = Intrinsic::PeInfo;
    else if (name() == "reserve_shared")       intrinsic_ = Intrinsic::Reserve;
    else if (name() == "atomic")               intrinsic_ = Intrinsic::Atomic;
    else if (name() == "cmpxchg")              intrinsic_ = Intrinsic::CmpXchg;
    else if (name() == "undef")                intrinsic_ = Intrinsic::Undef;
    else ELOG("unsupported thorin intrinsic");
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }

void Lam::app(const Def* callee, Defs args, Debug dbg) {
    app(callee, world().tuple(args), dbg);
}

void Lam::app(const Def* callee, const Def* arg, Debug dbg) {
    if (auto lam = callee->isa<Lam>()) {
        switch (lam->intrinsic()) {
            case Intrinsic::Branch: {
                auto cond = world().extract(arg, 0_s);
                auto t    = world().extract(arg, 1_s);
                auto f    = world().extract(arg, 2_s);
                if (auto lit = cond->isa<PrimLit>())
                    return app(lit->value().get_bool() ? t : f, Defs{}, dbg);
                if (t == f)
                    return app(t, Defs{}, dbg);
                if (is_not(cond))
                    return branch(cond->as<ArithOp>()->rhs(), f, t, dbg);
                break;
            }
            case Intrinsic::Match: {
                auto args = arg->as<Tuple>()->ops();
                if (args.size() == 2) return app(args[1], Defs{}, dbg);
                if (auto lit = args[0]->isa<PrimLit>()) {
                    for (size_t i = 2; i < args.size(); i++) {
                        if (world().extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return app(world().extract(args[i], 1), Defs{}, dbg);
                    }
                    return app(args[1], Defs{}, dbg);
                }
                break;
            }
            default:
                break;
        }
    }

    Def::update_op(1, world().app(callee, arg, dbg));
}

void Lam::branch(const Def* cond, const Def* t, const Def* f, Debug dbg) {
    return app(world().branch(), {cond, t, f}, dbg);
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

Lam::Lam(const Pi* pi, CC cc, Intrinsic intrinsic, Debug dbg)
    : Def(Node_Lam, pi, 2, dbg)
    , cc_(cc)
    , intrinsic_(intrinsic)
{
    set_op(0, world().literal_bool(false));
    set_op(1, world().top(pi->codomain()));

    contains_lam_ = true;
}

PrimType::PrimType(World& world, PrimTypeTag tag, size_t length, Debug dbg)
    : VectorType((int) tag, world.star(), Defs{}, length, dbg)
{}

MemType::MemType(World& world)
    : Def(Node_MemType, world.star(), Defs{}, {"mem"})
{}

FrameType::FrameType(World& world)
    : Def(Node_FrameType, world.star(), Defs{}, {"frame"})
{}

/*
 * hash
 */

uint64_t Def::vhash() const {
    if (is_nominal())
        return murmur3(gid());

    uint64_t seed = hash_combine(hash_begin((uint8_t) tag()), type()->gid());
    for (auto op : ops())
        seed = hash_combine(seed, op->gid());
    return seed;
}

uint64_t PtrType::vhash() const { return hash_combine(VectorType::vhash(), (uint64_t)device(), (uint64_t)addr_space()); }
uint64_t Var::vhash() const { return hash_combine(Def::vhash(), index()); }

/*
 * equal
 */

bool Def::equal(const Def* other) const {
    if (this->is_nominal() || other->is_nominal())
        return this == other;

    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->ops_[i] == other->ops_[i];
    return result;
}

bool Var::equal(const Def* other) const { return Def::equal(other) && this->index() == other->as<Var>()->index(); }

bool PtrType::equal(const Def* other) const {
    if (!VectorType::equal(other))
        return false;
    auto ptr = other->as<PtrType>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

/*
 * vrebuild
 */

const Def* Lam                ::vrebuild(World& to, const Def* t, Defs ops) const { assert(!is_nominal()); return to.lam(t->as<Pi>(), ops[0], ops[1], debug()); }
const Def* Sigma              ::vrebuild(World& to, const Def* t, Defs ops) const { assert(!is_nominal()); return to.sigma(t, ops, debug()); }
const Def* App                ::vrebuild(World& to, const Def*  , Defs ops) const { return to.app(ops[0], ops[1], debug()); }
const Def* DefiniteArrayType  ::vrebuild(World& to, const Def*  , Defs ops) const { return to.definite_array_type(ops[0], dim(), debug()); }
const Def* FrameType          ::vrebuild(World& to, const Def*  , Defs    ) const { return to.frame_type(); }
const Def* IndefiniteArrayType::vrebuild(World& to, const Def*  , Defs ops) const { return to.indefinite_array_type(ops[0], debug()); }
const Def* MemType            ::vrebuild(World& to, const Def*  , Defs    ) const { return to.mem_type(); }
const Def* Param              ::vrebuild(World& to, const Def*  , Defs ops) const { return to.param(ops[0]->as_lam(), debug()); }
const Def* Pi                 ::vrebuild(World& to, const Def*  , Defs ops) const { return to.pi(ops[0], ops[1], debug()); }
const Def* PrimType           ::vrebuild(World& to, const Def*  , Defs    ) const { return to.type(primtype_tag(), length(), debug()); }
const Def* PtrType            ::vrebuild(World& to, const Def*  , Defs ops) const { return to.ptr_type(ops.front(), length(), device(), addr_space()); }
const Def* Var                ::vrebuild(World& to, const Def* t, Defs    ) const { return to.var(t, index(), debug()); }
const Def* VariantType        ::vrebuild(World& to, const Def* t, Defs ops) const { return to.variant_type(t, ops, debug()); }

/*
 * vstub
 */

Lam*   Lam  ::vstub(World& to, const Def* type) const { assert(is_nominal()); return to.lam(type->as<Pi>(), cc(), intrinsic(), debug()); }
Sigma* Sigma::vstub(World& to, const Def* type) const { assert(is_nominal()); return to.sigma(type, num_ops(), debug()); }

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Def* type) {
   return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& Bottom             ::stream(std::ostream& os) const { return streamf(os, "{{⊥: {}}}", type()); }
std::ostream& DefiniteArrayType  ::stream(std::ostream& os) const { return streamf(os, "[{} x {}]", dim(), elem_type()); }
std::ostream& FrameType          ::stream(std::ostream& os) const { return os << "frame"; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[{}]", elem_type()); }
std::ostream& MemType            ::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& Var                ::stream(std::ostream& os) const { return streamf(os, "<{}:{}>", index(), type()); }
std::ostream& VariantType        ::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }

#if 0
std::ostream& Lam::stream(std::ostream& os) const {
    if (is_nominal())
    return streamf(os, "[{}].{}", name(), body());
}
#endif

std::ostream& Sigma::stream(std::ostream& os) const {
    if (is_nominal()) return os << unique_name();
    return stream_list(os, ops(), [&](const Def* type) { os << type; }, "[", "]");
}

std::ostream& Pi::stream(std::ostream& os) const {
    return is_cn()
        ? streamf(os, "cn {}", domain())
        : streamf(os, "Π{} -> {}", domain(), codomain());
}

std::ostream& PtrType::stream(std::ostream& os) const {
    if (is_vector())
        os << '<' << length() << " x ";
    os << pointee() << '*';
    if (is_vector())
        os << '>';
    if (device() != -1)
        os << '[' << device() << ']';
    switch (addr_space()) {
        case AddrSpace::Global:   os << "[Global]";   break;
        case AddrSpace::Texture:  os << "[Tex]";      break;
        case AddrSpace::Shared:   os << "[Shared]";   break;
        case AddrSpace::Constant: os << "[Constant]"; break;
        default: /* ignore unknown address space */      break;
    }
    return os;
}

std::ostream& PrimType::stream(std::ostream& os) const {
    if (is_vector())
        os << "<" << length() << " x ";

    switch (primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: os << #T; break;
#include "thorin/tables/primtypetable.h"
          default: THORIN_UNREACHABLE;
    }

    if (is_vector())
        os << ">";

    return os;
}

std::ostream& Def::stream(std::ostream& out) const { return out << unique_name(); }

std::ostream& Def::stream_assignment(std::ostream& os) const {
    return streamf(os, "{} {} = {} {}", type(), unique_name(), op_name(), stream_list(ops(), [&] (const Def* def) { os << def; })) << endl;
}
std::ostream& Lam::stream_head(std::ostream& os) const {
    os << unique_name();
    streamf(os, "{} {}", param()->type(), param());
    if (is_external())
        os << " extern ";
    if (cc() == CC::Device)
        os << " device ";
    if (filter())
        streamf(os, " @({})", filter());
    return os;
}

std::ostream& Lam::stream_body(std::ostream& os) const {
    return streamf(os, "{}", body());
}

void Lam::dump_head() const { stream_head(std::cout) << endl; }
void Lam::dump_body() const { stream_body(std::cout) << endl; }

}
