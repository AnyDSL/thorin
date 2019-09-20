#include "thorin/primop.h"

#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/util.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * constructors
 */

Known::Known(const Def* def, const Def* dbg)
    : Def(Node, rebuild, def->world().type_bool(), {def}, 0, dbg)
{}

//------------------------------------------------------------------------------

/*
 * rebuild
 */

// do not use any of d's type getters - during import we need to derive types from 't' in the new world 'to'

const Def* Alloc  ::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.alloc(thorin::as<Tag::Ptr>(t->as<Sigma>()->op(1))->arg(0), ops[0], dbg); }
const Def* Global ::rebuild(const Def* d, World& to, const Def*  , Defs ops, const Def* dbg) { return to.global(ops[0], ops[1], d->as<Global>()->is_mutable(), dbg); }
const Def* Hlt    ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.hlt(ops[0], dbg); }
const Def* Known  ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.known(ops[0], dbg); }
const Def* Run    ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.run(ops[0], dbg); }
const Def* Load   ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.load(ops[0], ops[1], dbg); }
const Def* Slot   ::rebuild(const Def*  , World& to, const Def* t, Defs ops, const Def* dbg) { return to.slot(thorin::as<Tag::Ptr>(t->as<Sigma>()->op(1))->arg(0), ops[0], dbg); }
const Def* Store  ::rebuild(const Def*  , World& to, const Def*  , Defs ops, const Def* dbg) { return to.store(ops[0], ops[1], ops[2], dbg); }

std::ostream& Global::stream(std::ostream& os) const { return os << unique_name(); }

}
