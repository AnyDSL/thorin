#include "anydsl/ref.h"

#include "anydsl/def.h"
#include "anydsl/cfg.h"
#include "anydsl/world.h"

namespace anydsl {

World& RVal::world() const { return load()->world(); }

World& VarRef::world() const { return var_->load()->world(); }
const Def* VarRef::load() const { return var_->load(); }
void VarRef::store(const Def* val) const { var_->store(val); }

const Def* TupleRef::load() const { 
    if (loaded_)
        return loaded_;

    return loaded_ = world().extract(lref_->load(), index_);
}

void TupleRef::store(const Def* val) const { 
    lref_->store(world().insert(lref_->load(), index_, val)); 
}

World& TupleRef::world() const { 
    return loaded_ ? loaded_->world() : lref_->world(); 
}

} // namespace anydsl
