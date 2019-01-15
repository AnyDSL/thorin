#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lam.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * misc
 */

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

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

const Def* Lam                ::vrebuild(World& to, const Def* t, Defs ops) const { assert(!is_nominal()); return to.lam(t->as<Pi>(), ops[0], ops[1], debug()); }
const Def* Sigma              ::vrebuild(World& to, const Def* t, Defs ops) const { assert(!is_nominal()); return to.sigma(t, ops, debug()); }
const Def* App                ::vrebuild(World& to, const Def*  , Defs ops) const { return to.app(ops[0], ops[1], debug()); }
const Def* VariantType        ::vrebuild(World& to, const Def* t, Defs ops) const { return to.variant_type(t, ops, debug()); }
const Def* Var                ::vrebuild(World& to, const Def* t, Defs    ) const { return to.var(t, index(), debug()); }
const Def* DefiniteArrayType  ::vrebuild(World& to, const Def*  , Defs ops) const { return to.definite_array_type(ops[0], dim(), debug()); }
const Def* Pi                 ::vrebuild(World& to, const Def*  , Defs ops) const { return to.pi(ops[0], ops[1], debug()); }
const Def* FrameType          ::vrebuild(World& to, const Def*  , Defs    ) const { return to.frame_type(); }
const Def* IndefiniteArrayType::vrebuild(World& to, const Def*  , Defs ops) const { return to.indefinite_array_type(ops[0], debug()); }
const Def* MemType            ::vrebuild(World& to, const Def*  , Defs    ) const { return to.mem_type(); }
const Def* PrimType           ::vrebuild(World& to, const Def*  , Defs    ) const { return to.type(primtype_tag(), length(), debug()); }

const Def* PtrType::vrebuild(World& to, const Def*, Defs ops) const {
    return to.ptr_type(ops.front(), length(), device(), addr_space());
}

//------------------------------------------------------------------------------

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return world().ptr_type(ptr->pointee());
    return world().type(as<PrimType>()->primtype_tag());
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
 * hash
 */

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device(), (uint64_t)addr_space());
}

uint64_t Var::vhash() const { return hash_combine(Def::vhash(), index()); }

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Var::equal(const Def* other) const { return Def::equal(other) && this->index() == other->as<Var>()->index(); }

bool PtrType::equal(const Def* other) const {
    if (!VectorType::equal(other))
        return false;
    auto ptr = other->as<PtrType>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

//------------------------------------------------------------------------------

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Def* type) {
   return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "(", ")");
}

std::ostream& BottomType         ::stream(std::ostream& os) const { return os << "bottom_type"; }
std::ostream& DefiniteArrayType  ::stream(std::ostream& os) const { return streamf(os, "[{} x {}]", dim(), elem_type()); }
std::ostream& FrameType          ::stream(std::ostream& os) const { return os << "frame"; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[{}]", elem_type()); }
std::ostream& MemType            ::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& Var                ::stream(std::ostream& os) const { return streamf(os, "<{}:{}>", index(), type()); }
std::ostream& VariantType        ::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }

std::ostream& Lam::stream(std::ostream& os) const {
    if (is_nominal())
    return streamf(os, "[{}].{}", name(), body());
}

std::ostream& Sigma::stream(std::ostream& os) const {
    if (is_nominal()) return os << unique_name();
    return stream_list(os, type->ops(), [&](const Def* type) { os << type; }, "[", "]");
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

//------------------------------------------------------------------------------

TypeTable::TypeTable()
    : unit_ (unify(new TupleType(*this, Types())))
    , bottom_type_ (unify(new BottomType (*this)))
    , cn0_  (unify(new Pi   (*this, unit_, bottom_type_)))
    , mem_  (unify(new MemType  (*this)))
    , frame_(unify(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) \
    , T##_(unify(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

const StructType* TypeTable::struct_type(Symbol name, size_t size) {
    auto type = new StructType(*this, name, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

const Type* TypeTable::app_(const Type* callee, const Type* op) {
    auto app = unify(new App_(*this, callee, op));

    if (auto cache = app->cache_)
        return cache;
    if (auto lambda = app->callee()->template isa<Lambda>()) {
        Type2Type map;
        return app->cache_ = lambda->body()->reduce(1, op, map);
    } else {
        return app->cache_ = app;
    }

    return app;
}

//------------------------------------------------------------------------------

}
