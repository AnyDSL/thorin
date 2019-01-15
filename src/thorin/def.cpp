#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lam.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Def::gid_counter_ = 1;

uint64_t Def::vhash() const {
    if (is_nominal())
        return murmur3(gid());

    uint64_t seed = hash_combine(hash_begin((uint8_t) tag()), type()->gid());
    for (auto op : ops())
        seed = hash_combine(seed, op->gid());
    return seed;
}

bool Def::equal(const Def* other) const {
    if (this->is_nominal() || other->is_nominal())
        return this == other;

    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->ops_[i] == other->ops_[i];
    return result;
}

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
std::ostream& Def::stream(std::ostream& out) const { return out << unique_name(); }

std::ostream& Def::stream_assignment(std::ostream& os) const {
    return streamf(os, "{} {} = {} {}", type(), unique_name(), op_name(), stream_list(ops(), [&] (const Def* def) { os << def; })) << endl;
}

#if THORIN_ENABLE_CHECKS
void force_use_dump() {
    Defs defs;
    defs.dump();
    Array<const Def*> a;
    a.dump();
}
#endif
}
