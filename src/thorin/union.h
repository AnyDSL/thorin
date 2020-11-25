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

/// Matches against a value, using @p ptrns.
class Match : public Def {
private:
    Match(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* arg() const { return op(0); }
    Defs ptrns() const { return ops().skip_front(); }
    const Def* ptrn(size_t i) const { return op(i + 1); }
    size_t num_ptrns() const { return num_ops() - 1; }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Match;
    friend class World;
};

/// Pattern type.
class Case : public Def {
private:
    Case(const Def* type, const Def* domain, const Def* codomain, const Def* dbg)
        : Def(Node, type, {domain, codomain}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* domain() const { return op(0); }
    const Def* codomain() const { return op(1); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Case;
    friend class World;
};

/// Pattern value.
class Ptrn : public Def {
private:
    Ptrn(const Def* type, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    Ptrn* set(const Def* matcher, const Def* body) { return Def::set({matcher, body})->as<Ptrn>(); }
    const Def* matcher() const { return op(0); }
    const Def* body() const { return op(1); }
    /// @name type
    //@{
    const Case* type() const { return Def::type()->as<Case>(); }
    const Def*  domain() const { return type()->domain(); }
    const Def*  codomain() const { return type()->codomain(); }
    //@}
    /// @name misc getters
    //@{
    bool is_trivial() const;
    bool matches(const Def*) const;
    //@}
    /// @name virtual methods
    //@{
    Ptrn* stub(World&, const Def*, const Def*) override;
    //@}

    static constexpr auto Node = Node::Ptrn;
    friend class World;
};

}

#endif
