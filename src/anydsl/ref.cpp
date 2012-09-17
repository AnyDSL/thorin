#include "anydsl/ref.h"

#include "anydsl/def.h"
#include "anydsl/cfg.h"
#include "anydsl/world.h"

namespace anydsl {

RefPtr Ref::create(const Def* def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create(Var* var) { return RefPtr(new VarRef(var)); }
RefPtr Ref::create(RefPtr lref, u32 index) { return RefPtr(new TupleRef(lref, index)); }

World& RVal::world() const { return load()->world(); }

World& VarRef::world() const { return var_->load()->world(); }
const Def* VarRef::load() const { return var_->load(); }
void VarRef::store(const Def* val) const { var_->store(val); }

const Def* TupleRef::load() const { return world().extract(lref_->load(), index_); }
void TupleRef::store(const Def* val) const { lref_->store(world().insert(lref_->load(), index_, val)); }

} // namespace anydsl
