#include "anydsl2/ref.h"

#include "anydsl2/def.h"
#include "anydsl2/irbuilder.h"
#include "anydsl2/world.h"

namespace anydsl2 {

World& RVal::world() const { return load()->world(); }

World& VarRef::world() const { return type_->world(); }
const Def* VarRef::load() const { return bb_->get_value(handle_, type_); }
void VarRef::store(const Def* def) const { bb_->set_value(handle_, def); }

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

} // namespace anydsl2
