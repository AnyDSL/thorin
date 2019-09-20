#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/util.h"

namespace thorin {

//------------------------------------------------------------------------------

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 * @deprecated
 */
class Global : public Def {
private:
    Global(const Def* type, const Def* id, const Def* init, bool is_mutable, const Def* dbg)
        : Def(Node, rebuild, type, {id, init}, is_mutable, dbg)
    {}

public:
    /// This thing's sole purpose is to differentiate on global from another.
    const Def* id() const { return op(0); }
    const Def* init() const { return op(1); }
    bool is_mutable() const { return fields(); }
    const App* type() const { return thorin::as<Tag::Ptr>(Def::type()); }
    const Def* alloced_type() const { return type()->arg(0); }

    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Global;
    friend class World;
};

}

#endif
