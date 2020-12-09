#ifndef THORIN_SET_H
#define THORIN_SET_H

#include "thorin/def.h"

namespace thorin {

class Lam;

template<bool up>
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
    size_t find(const Def* type) const;
    const Def* get(const Def* type) const { return op(find(type)); }

    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Bound* stub(World&, const Def*, const Def*) override;
    //@}

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
 * Tests the @p value of type @p Join whether it currently holds @em type @p index.
 * Note, that @p index is a @em type!
 * Yields @p match if @c true and @p clash otherwise.
 * @p match must be of type <tt> A -> B </tt>.
 * @p clash must be of type <tt> [A, index] -> C </tt>.
 * This operation is usually known as @c case but @c case is a keyword in C++, so we call it @p Test.
 */
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

/// Ext%remum. Either @p Top (@p up) or @p Bot%tom.
template<bool up>
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

    static constexpr auto Node = up ? Node::Top : Node::Bot;
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

inline bool is_ext(bool top, const Def* def) {
    if (auto ext = isa_ext(def)) return *ext == top;
    return false;
}

inline std::optional<bool> isa_bound(const Def* def) {
    if (def->isa<Meet>()) return false;
    if (def->isa<Join>()) return true;
    return {};
}

inline bool is_bound(bool join, const Def* def) {
    if (auto ext = isa_bound(def)) return *ext == join;
    return false;
}

}

#endif
