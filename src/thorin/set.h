#ifndef THORIN_UNION_H
#define THORIN_UNION_H

#include "thorin/def.h"

namespace thorin {

class Union : public Def {
private:
    /// Constructor for a @em structural Union.
    Union(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Union.
    Union(const Def* type, size_t size, const Def* dbg)
        : Def(Node, type, size, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Union* stub(World&, const Def*, const Def*) override;
    //@}

    static constexpr auto Node = Node::Union;
    friend class World;
};

class Which : public Def {
private:
    Which(const Def* type, const Def* value, const Def* dbg)
        : Def(Node, type, {value}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* value() const { return op(0); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Which;
    friend class World;
};

}

#endif
