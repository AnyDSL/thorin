#include "anydsl/ref.h"

#include "anydsl/def.h"
#include "anydsl/cfg.h"
#include "anydsl/world.h"

namespace anydsl {

World& RVal::world() const { return load()->world(); }
World& VarRef::world() const { return var()->load()->world(); }

const Def* VarRef::load() const { return var()->load(); }
void VarRef::store(const Def* val) const { var()->store(val); }

const Def* TupleRef::load() const { 
    return world().extract(lref()->load(), index());
}

void TupleRef::store(const Def* val) const { 
    lref()->store(world().insert(lref()->load(), index(), val));
}

} // namespace anydsl
