#include "thorin/def.h"

#include <algorithm>
#include <stack>

#include "thorin/rewrite.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/utility.h"

namespace thorin {

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
    , dbg_(dbg)
    , type_(type)
{
    gid_ = world().next_gid();
    std::copy(ops.begin(), ops.end(), ops_ptr());

    if (node == Node::Space) {
        hash_ = murmur3(gid());
    } else {
        hash_ = type->gid();
        for (auto op : ops)
            hash_ = murmur3(hash_, u32(op->gid()));
        hash_ = murmur3(hash_, fields_);
        hash_ = murmur3_rest(hash_, u8(node));
        hash_ = murmur3_finalize(hash_, (num_ops() + 1) * sizeof(u32) + 1);
    }
}

Def::Def(node_t node, const Def* type, size_t num_ops, uint64_t fields, const Def* dbg)
    : fields_(fields)
    , node_(node)
    , nominal_(true)
    , const_(false)
    , order_(0)
    , num_ops_(num_ops)
    , dbg_(dbg)
    , type_(type)
{
    gid_ = world().next_gid();
    hash_ = murmur3(gid());
    std::fill_n(ops_ptr(), num_ops, nullptr);
    if (!type->is_const()) type->uses_.emplace(this, -1);
}

Kind::Kind(World& world)
    : Def(Node, (const Def*) world.space(), Defs{}, 0, nullptr)
{}

Nat::Nat(World& world)
    : Def(Node, world.kind(), Defs{}, 0, nullptr)
{}

/*
 * rebuild
 */

const Def* App    ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.app(o[0], o[1], dbg); }
const Def* Arr    ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.arr(o[0], o[1], dbg); }
const Def* Axiom  ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.axiom(normalizer(), t, tag(), flags(), dbg); }
const Def* Bot    ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.bot(t, dbg); }
const Def* Case   ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.case_(o[0], o[1], dbg); }
const Def* Extract::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.extract_(t, o[0], o[1], dbg); }
const Def* Global ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.global(o[0], o[1], is_mutable(), dbg); }
const Def* Insert ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.insert(o[0], o[1], o[2], dbg); }
const Def* Kind   ::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.kind(); }
const Def* Lam    ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.lam(t->as<Pi>(), o[0], o[1], dbg); }
const Def* Lit    ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.lit(t, get(), dbg); }
const Def* Match  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.match(o[0], o.skip_front(), dbg); }
const Def* Nat    ::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.type_nat(); }
const Def* Pack   ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.pack(t->arity(), o[0], dbg); }
const Def* Param  ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.param(t, o[0]->as_nominal(), dbg); }
const Def* Pi     ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.pi(o[0], o[1], dbg); }
const Def* Proxy  ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.proxy(t, o, as<Proxy>()->index(), as<Proxy>()->flags(), dbg); }
const Def* Sigma  ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.sigma(t, o, dbg); }
const Def* Space  ::rebuild(World& w, const Def*  , Defs  , const Def*    ) const { return w.space(); }
const Def* Top    ::rebuild(World& w, const Def* t, Defs  , const Def* dbg) const { return w.top(t, dbg); }
const Def* Tuple  ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.tuple(t, o, dbg); }
const Def* Union  ::rebuild(World& w, const Def* t, Defs o, const Def* dbg) const { return w.union_(t, o, dbg); }
const Def* Which  ::rebuild(World& w, const Def*  , Defs o, const Def* dbg) const { return w.which(o[0], dbg); }

/*
 * stub
 */

Lam*   Lam  ::stub(World& w, const Def* t, const Def* dbg) { return w.nom_lam  (t->as<Pi>(), cc(), intrinsic(), dbg); }
Pi*    Pi   ::stub(World& w, const Def* t, const Def* dbg) { return w.nom_pi   (t, dbg); }
Ptrn*  Ptrn ::stub(World& w, const Def* t, const Def* dbg) { return w.nom_ptrn (t->as<Case>(), dbg); }
Sigma* Sigma::stub(World& w, const Def* t, const Def* dbg) { return w.nom_sigma(t, num_ops(), dbg); }
Union* Union::stub(World& w, const Def* t, const Def* dbg) { return w.nom_union(t, num_ops(), dbg); }
Arr*   Arr  ::stub(World& w, const Def* t, const Def* dbg) { return w.nom_arr  (t, shape(), dbg); }

/*
 * restructure
 */

const Pi* Pi::restructure() {
    if (!is_free(param(), codomain())) return world().pi(domain(), codomain(), dbg());
    return nullptr;
}

const Def* Arr::restructure() {
    auto& w = world();
    if (auto n = isa_lit(shape()))
        return w.sigma(type(), Array<const Def*>(*n, [&](size_t i) { return apply(w.lit_int(*n, i)).back(); }));
    return nullptr;
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
    if (isa<Space>()) return Defs();

    size_t offset = dbg() ? 2 : 1;
    return Defs((is_set() ? num_ops_ : 0) + offset, ops_ptr() - offset);
}

const Param* Def::param(const Def* dbg) {
    auto& w = world();
    if (auto lam    = isa<Lam  >()) return w.param(lam ->domain(), lam,    dbg);
    if (auto ptrn   = isa<Ptrn >()) return w.param(ptrn->domain(), ptrn,   dbg);
    if (auto pi     = isa<Pi   >()) return w.param(pi  ->domain(), pi,     dbg);
    if (auto sigma  = isa<Sigma>()) return w.param(sigma,          sigma,  dbg);
    if (auto union_ = isa<Union>()) return w.param(union_,         union_, dbg);
    if (auto arr    = isa<Arr  >()) return w.param(w.type_int(arr ->shape()), arr,  dbg); // TODO shapes like (2, 3)
    if (auto pack   = isa<Pack >()) return w.param(w.type_int(pack->shape()), pack, dbg); // TODO shapes like (2, 3)
    THORIN_UNREACHABLE;
}

const Param* Def::param() { return param(nullptr); }
const Def*   Def::param(size_t i) { return param(i, nullptr); }
size_t       Def::num_params() { return param()->num_outs(); }

Sort Def::level() const {
    if (                        isa<Space>()) return Sort::Space;
    if (                type()->isa<Space>()) return Sort::Kind;
    if (        type()->type()->isa<Space>()) return Sort::Type;
    return Sort::Term;
}

Sort Def::sort() const {
    switch (node()) {
        case Node::Space: return Sort::Space;
        case Node::Kind:  return Sort::Kind;
        case Node::Arr:
        case Node::Case:
        case Node::Nat:
        case Node::Pi:
        case Node::Sigma:
        case Node::Union: return Sort::Type;
        case Node::Global:
        case Node::Insert:
        case Node::Lam:
        case Node::Pack:
        case Node::Ptrn:
        case Node::Tuple:
        case Node::Which: return Sort::Term;
        default:          return Sort(int(type()->sort()) - 1);
    }
}

const Def* Def::tuple_arity() const {
    if (auto sigma  = isa<Sigma>()) return world().lit_nat(sigma->num_ops());
    if (auto arr    = isa<Arr  >()) return arr->shape();
    if (sort() == Sort::Term)       return type()->tuple_arity();
    assert(sort() == Sort::Type);
    return world().lit_nat(1);
}

const Def* Def::arity() const {
    if (auto sigma  = isa<Sigma>()) return world().lit_nat(sigma->num_ops());
    if (auto union_ = isa<Union>()) return world().lit_nat(union_->num_ops());
    if (auto arr    = isa<Arr  >()) return arr->shape();
    if (sort() == Sort::Term)       return type()->arity();
    return world().lit_nat(1);
}

bool Def::equal(const Def* other) const {
    if (isa<Space>() || this->isa_nominal() || other->isa_nominal())
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
        return dbg() ? w.insert(dbg(), 3_s, 0_s, w.tuple_str(unique_name())) : w.tuple_str(unique_name());
#endif
    return dbg();
}

void Def::set_name(const std::string& n) const {
    auto& w = world();
    auto name = w.tuple_str(n);

    if (dbg_ == nullptr) {
        auto file = w.tuple_str("");
        auto begin = w.lit_nat(nat_t(-1));
        auto finis = w.lit_nat(nat_t(-1));
        auto meta = w.bot(w.bot_kind());
        dbg_ = w.tuple({name, w.tuple({file, begin, finis}), meta});
    } else {
        dbg_ = w.insert(dbg_, 3_s, 0_s, name);
    }
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

    if (!isa<Space>()) {
        if (!type()->is_const()) {
            const_ = false;
            const auto& p = type()->uses_.emplace(this, -1);
            assert_unused(p.second);
        }
    }

    if (dbg()) const_ &= dbg()->is_const();
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

std::string Def::unique_name() const { return (isa_nominal() ? std::string{} : std::string{"%"}) + debug().name + "_" + std::to_string(gid()); }

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
            def = callee != app->callee() ? world().app(callee, app->arg(), app->dbg()) : app;
            break;
        }
    }
    return def;
}

const Def* Def::refine(size_t i, const Def* new_op) const {
    Array<const Def*> new_ops(ops());
    new_ops[i] = new_op;
    return rebuild(world(), type(), new_ops, dbg());
}

/*
 * Global
 */

const App* Global::type() const { return thorin::as<Tag::Ptr>(Def::type()); }
const Def* Global::alloced_type() const { return type()->arg(0); }

}
