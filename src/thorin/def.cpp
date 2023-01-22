#include "thorin/def.h"

#include <algorithm>
#include <stack>

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Def::gid_counter_ = 1;

Def::Def(NodeTag tag, World& world, const Type* type, Defs ops, Debug dbg)
    : tag_(tag)
    , ops_(ops.size())
    , world_(world)
    , type_(type)
    , debug_(dbg)
    , gid_(gid_counter_++)
    , nom_(false)
    , dep_(tag == Node_Continuation ? Dep::Cont  :
           tag == Node_Param        ? Dep::Param :
                                      Dep::Bot   )
{
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        set_op(i, ops[i]);
}

Def::Def(NodeTag tag, World& world, const Type* type, size_t size, Debug dbg)
    : tag_(tag)
    , ops_(size)
    , world_(world)
    , type_(type)
    , debug_(dbg)
    , gid_(gid_counter_++)
    , nom_(true)
    , dep_(tag == Node_Continuation ? Dep::Cont  :
           tag == Node_Param        ? Dep::Param :
                                      Dep::Bot   )
{}

Debug Def::debug_history() const {
#if THORIN_ENABLE_CHECKS
    return world().track_history() ? Debug(unique_name(), loc()) : debug();
#else
    return debug();
#endif
}

void Def::set_name(const std::string& name) const { debug_.name = name; }

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(def && "setting null pointer");
    assert(&def->world() == &world());
    ops_[i] = def;
    // A Param/Continuation should not have other bits than its own set.
    // (Right now, Param doesn't have ops, but this will change in the future).
    if (!isa_nom<Continuation>() && !isa<Param>())
        dep_ |= def->dep(); // what about unset op then ? and cascading uses ?
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
    // Note: if replace() didn't touch the uses, we could assert for nominalness here !
    assert(ops_[i] && "must be set");
    unregister_use(i);
    ops_[i] = nullptr;
}

void Def::unset_ops() {
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        unset_op(i);
}

std::string Def::unique_name() const {
    return name() + "_" + std::to_string(gid());
}

bool is_unit(const Def* def) {
    return def->type() == def->world().unit();
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

void Def::replace_uses(const Def* with) const {
    world().DLOG("replace uses: {} -> {}", this, with);
    if (this != with) {
        assert(&with->world() == &this->world());
        for (auto& use : copy_uses()) {
            auto def = const_cast<Def*>(use.def());
            auto index = use.index();
            def->unset_op(index);
            def->set_op(index, with);
        }

        uses_.clear();
    }
}

World& Def::world() const { return world_; }

uint64_t UseHash::hash(Use use) {
    assert(use->gid() != uint32_t(-1));
    hash_t seed = hash_begin(use.index());
    seed = hash_combine(seed, uint32_t(use->gid()));
    return seed;
}

}
