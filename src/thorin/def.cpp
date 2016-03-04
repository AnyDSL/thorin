#include "thorin/def.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/queue.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Def::gid_counter_ = 1;

void Def::set_op(size_t i, const Def* def) {
    assert(!op(i) && "already set");
    assert(def && "setting null pointer");
    ops_[i] = def;
    assert(!def->uses_.contains(Use(i, this)));
    const auto& p = def->uses_.emplace(i, this);
    assert_unused(p.second);
}

void Def::unregister_uses() const {
    for (size_t i = 0, e = size(); i != e; ++i)
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
    for (size_t i = 0, e = size(); i != e; ++i)
        unset_op(i);
}

std::string Def::unique_name() const {
    std::ostringstream oss;
    oss << name << '_' << gid();
    return oss.str();
}

bool Def::is_const() const {
    if (isa<Param>()) return false;
    if (isa<PrimOp>()) {
        for (auto op : ops()) { // TODO slow because ops form a DAG not a tree
            if (!op->is_const())
                return false;
        }
    }

    return true; // lambdas are always const
}

bool Def::is_primlit(int val) const {
    if (auto lit = this->isa<PrimLit>()) {
        switch (lit->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit->value().get_##T() == T(val);
#include "thorin/tables/primtypetable.h"
        }
    }

    if (auto vector = this->isa<Vector>()) {
        for (auto op : vector->ops()) {
            if (!op->is_primlit(val))
                return false;
        }
        return true;
    }
    return false;
}

bool Def::is_minus_zero() const {
    if (auto lit = this->isa<PrimLit>()) {
        Box box = lit->value();
        switch (lit->primtype_kind()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return box.get_##T() == T(0);
#define THORIN_F_TYPE(T, M) case PrimType_##T: return box.get_##T() == T(-0.0);
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }
    }
    return false;
}

void Def::replace(const Def* with) const {
    DLOG("replace: % -> %", this, with);
    assert(type() == with->type());
    if (this != with) {
        std::queue<const PrimOp*> queue;

        auto enqueue = [&](const Def* def) {
            if (auto primop = def->isa<PrimOp>()) {
                if (!primop->is_outdated()) {
                    queue.push(primop);
                    primop->is_outdated_ = true;
                }
            }
        };

        for (auto use : uses()) {
            const_cast<Def*>(use.def())->unset_op(use.index());
            const_cast<Def*>(use.def())->set_op(use.index(), with);
            enqueue(use);
        }

        while (!queue.empty()) {
            for (auto use : pop(queue)->uses())
                enqueue(use);
        }

        auto& this_trackers = world().trackers(this);
        auto& with_trackers = world().trackers(with);
        for (auto tracker : this_trackers) {
            tracker->def_ = with;
            with_trackers.emplace(tracker);
        }

        uses_.clear();
        this_trackers.clear();
    }
}

void Def::dump() const {
    auto primop = this->isa<PrimOp>();
    if (primop && !primop->is_const())
        primop->stream_assignment(std::cout);
    else {
        std::cout << this;
        std::cout << std::endl;
    }
}

// this workaround is needed in order to disambiguate def->op(0) from def->op(uint64_t) vs def->op(const Def*)
template<class T> const Def* Def::op(const T* def) const { return op(def->template primlit_value<uint64_t>()); }
template const Def* Def::op<Def>(const Def*) const;         // instantiate method
template const Def* Def::op<PrimLit>(const PrimLit*) const; // instantiate method

World& Def::world() const { return type()->world(); }
Lambda* Def::as_lambda() const { return const_cast<Lambda*>(scast<Lambda>(this)); }
Lambda* Def::isa_lambda() const { return const_cast<Lambda*>(dcast<Lambda>(this)); }
int Def::order() const { return type()->order(); }
size_t Def::length() const { return type().as<VectorType>()->length(); }
std::ostream& Def::stream(std::ostream& out) const { return out << unique_name(); }

std::ostream& operator << (std::ostream& os, const Def* def) { return def->stream(os); }
std::ostream& operator << (std::ostream& os, Use use) { return use->stream(os); }

HashSet<Tracker*>& Tracker::trackers(const Def* def) { return def->world().trackers_[def]; }

}
