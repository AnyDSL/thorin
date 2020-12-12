#ifndef THORIN_LATTICE_H
#define THORIN_LATTICE_H

#include "thorin/def.h"

namespace thorin {

class Lam;
class Sigma;

class Bound : public Def {
protected:
    /// Constructor for a @em structural Bound.
    Bound(node_t node, const Def* type, Defs ops, const Def* dbg)
        : Def(node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Bound.
    Bound(node_t node, const Def* type, size_t size, const Def* dbg)
        : Def(node, type, size, 0, dbg)
    {}

public:
    size_t find(const Def* type) const;
    const Lit* index(const Def* type) const;
    const Def* get(const Def* type) const { return op(find(type)); }
    const Sigma* convert() const;
};

template<bool up>
class TBound : public Bound {
private:
    /// Constructor for a @em structural Bound.
    TBound(const Def* type, Defs ops, const Def* dbg)
        : Bound(Node, type, ops, dbg)
    {}
    /// Constructor for a @em nominal Bound.
    TBound(const Def* type, size_t size, const Def* dbg)
        : Bound(Node, type, size, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    TBound* stub(World&, const Def*, const Def*) override;
    //@}

    const Sigma* convert() const;

    static constexpr auto Node = up ? Node::Join : Node::Meet;
    friend class World;
};

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

/**
 * Tests the @p value of type @p Join whether it currently holds @em type @p probe.
 * Note, that @p probe is a @em type!
 * Yields @p match if @c true and @p clash otherwise.
 * @p match must be of type <tt> A -> B </tt>.
 * @p clash must be of type <tt> [A, probe] -> C </tt>.
 * This operation is usually known as @c case but @c case is a keyword in C++, so we call it @p Test.
 */
class Test : public Def {
private:
    Test(const Def* type, const Def* value, const Def* probe, const Def* match, const Def* clash, const Def* dbg)
        : Def(Node, type, {value, probe, match, clash}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* value() const { return op(0); }
    const Def* probe() const { return op(1); }
    const Def* match() const { return op(2); }
    const Def* clash() const { return op(3); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Test;
    friend class World;
};

/// Common base for @p Ext%remum.
class Ext : public Def {
protected:
    Ext(node_t node, const Def* type, const Def* dbg)
        : Def(node, type, Defs{}, 0, dbg)
    {}
};

/// Ext%remum. Either @p Top (@p up) or @p Bot%tom.
template<bool up>
class TExt : public Ext {
private:
    TExt(const Def* type, const Def* dbg)
        : Ext(Node, type, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = up ? Node::Top : Node::Bot;
    friend class World;
};

using Bot  = TExt<false>;
using Top  = TExt<true >;
using Meet = TBound<false>;
using Join = TBound<true >;

inline const Ext* isa_ext(const Def* def) {
    return def->isa<Bot>() || def->isa<Top>() ? static_cast<const Ext*>(def) : nullptr;
}

inline const Bound* isa_bound(const Def* def) {
    return def->isa<Meet>() || def->isa<Join>() ? static_cast<const Bound*>(def) : nullptr;
}

}

#endif
