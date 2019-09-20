#include "thorin/primop.h"

#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/util.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * rebuild
 */

// do not use any of d's type getters - during import we need to derive types from 't' in the new world 'to'

const Def* Global ::rebuild(const Def* d, World& to, const Def*  , Defs ops, const Def* dbg) { return to.global(ops[0], ops[1], d->as<Global>()->is_mutable(), dbg); }

std::ostream& Global::stream(std::ostream& os) const { return os << unique_name(); }

}
