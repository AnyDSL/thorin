#ifndef THORIN_UNION_H
#define THORIN_UNION_H

#include "thorin/def.h"

namespace thorin {

class Lam;

/// Ext%tremum. Either @p Top (@p up) or @p Bot%tom.
template<bool up_>
class Ext : public Def {
private:
    Ext(const Def* type, const Def* dbg)
        : Def(Node, type, Defs{}, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr bool up = up_;
    static constexpr auto Node = up_ ? Node::Top : Node::Bot;
    friend class World;
};

template<bool up_>
class Bound : public Def {
private:
    /// Constructor for a @em structural Bound.
    Bound(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Bound.
    Bound(const Def* type, size_t size, const Def* dbg)
        : Def(Node, type, size, 0, dbg)
    {}

public:
    bool contains(const Def* type) const;

    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Bound* stub(World&, const Def*, const Def*) override;
    //@}

    static constexpr bool up = up_;
    static constexpr auto Node = up ? Node::Join : Node::Meet;
    friend class World;
};

using Bot  = Ext<false>;
using Top  = Ext<true >;
using Meet = Bound<false>;
using Join = Bound<true >;

inline std::optional<bool> isa_ext(const Def* def) {
    if (def->isa<Bot>()) return false;
    if (def->isa<Top>()) return true;
    return {};
}

inline std::optional<bool> isa_bound(const Def* def) {
    if (def->isa<Meet>()) return false;
    if (def->isa<Join>()) return true;
    return {};
}

/// Constructs a @p Meet value.
class Et : public Def {
private:
    Et(const Def* type, Defs defs, const Def* dbg)
        : Def(Node, type, defs, 0, dbg)
    {}

public:
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Et;
    friend class World;
};

/// Constructs a @p Join value.
class Vel : public Def {
private:
    Vel(const Def* type, const Def* value, const Def* dbg)
        : Def(Node, type, {value}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* value() const { return op(0); }
    //@}
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Vel;
    friend class World;
};

/// Picks the aspect of @p type of a @p Meet or @p Join @p value.
class Pick : public Def {
private:
    Pick(const Def* type, const Def* value, const Def* dbg)
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

    static constexpr auto Node = Node::Pick;
    friend class World;
};

/// Retuns the @p type held by @p value to index using @p Pick.
class Test : public Def {
private:
    Test(const Def* type, const Def* value, const Def* index, const Def* match, const Def* clash, const Def* dbg)
        : Def(Node, type, {value, index, match, clash}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* value() const { return op(0); }
    const Def* index() const { return op(1); }
    const Lam* match() const;
    const Lam* clash() const;
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Test;
    friend class World;
};

}

#endif
