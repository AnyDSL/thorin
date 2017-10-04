#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Def::gid_counter_ = 1;

Def::Def(NodeTag tag, const Type* type, size_t size, Debug dbg)
    : tag_(tag)
    , ops_(size)
    , type_(type)
    , gid_(gid_counter_++)
    , debug_(dbg)
{}

Debug Def::debug_history() const {
#ifndef NDEBUG
    return world().track_history() ? Debug(location(), unique_name()) : debug();
#else
    return debug();
#endif
}

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(def && "setting null pointer");
    ops_[i] = def;
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

void Def::unset_ops() {
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        unset_op(i);
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
    if (def->isa<Param>()) return false;
    if (def->isa<PrimOp>()) {
        for (auto op : def->ops()) { // TODO slow because ops form a DAG not a tree
            if (!is_const(op))
                return false;
        }
    }

    return true; // continuations are always const
}

size_t vector_length(const Def* def) { return def->type()->as<VectorType>()->length(); }

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
            def->unset_op(index);
            def->set_op(index, with);
        }

        uses_.clear();
        substitute_ = with;
    }
}

void Def::dump() const {
    auto primop = this->isa<PrimOp>();
    if (primop && !is_const(primop))
        primop->stream_assignment(std::cout);
    else {
        std::cout << this;
        std::cout << std::endl;
    }
}

World& Def::world() const { return *static_cast<World*>(&type()->table()); }
Continuation* Def::as_continuation() const { return const_cast<Continuation*>(scast<Continuation>(this)); }
Continuation* Def::isa_continuation() const { return const_cast<Continuation*>(dcast<Continuation>(this)); }
std::ostream& Def::stream(std::ostream& out) const { return out << unique_name(); }

}
