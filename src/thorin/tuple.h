#ifndef THORIN_TUPLE_H
#define THORIN_TUPLE_H

#include "thorin/def.h"

namespace thorin {

class Sigma : public Def {
private:
    /// Constructor for a @em structural Sigma.
    Sigma(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Sigma.
    Sigma(const Def* type, size_t size, const Def* dbg)
        : Def(Node, type, size, 0, dbg)
    {}

public:
    /// @name setters
    //@{
    Sigma* set(size_t i, const Def* def) { return Def::set(i, def)->as<Sigma>(); }
    Sigma* set(Defs ops) { return Def::set(ops)->as<Sigma>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Sigma* stub(World&, const Def*, const Def*) override;
    //@}

    static constexpr auto Node = Node::Sigma;
    friend class World;
};

/// Data constructor for a @p Sigma.
class Tuple : public Def {
private:
    Tuple(const Def* type, Defs args, const Def* dbg)
        : Def(Node, type, args, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Tuple;
    friend class World;
};

class Arr : public Def {
private:
    /// Constructor for a @em structural Arr.
    Arr(const Def* type, const Def* shape, const Def* body, const Def* dbg)
        : Def(Node, type, {shape, body}, 0, dbg)
    {}
    /// Constructor for a @em nominal Arr.
    Arr(const Def* type, const Def* shape, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {
        Def::set(0, shape);
    }

public:
    /// @name ops
    //@{
    const Def* shape() const { return op(0); }
    const Def* body() const { return op(1); }
    //@}
    /// @name methods for nominals
    //@{
    Arr* set(const Def* body) { return Def::set(1, body)->as<Arr>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Arr* stub(World&, const Def*, const Def*) override;
    const Def* restructure();
    //@}

    static constexpr auto Node = Node::Arr;
    friend class World;
};

class Pack : public Def {
private:
    Pack(const Def* type, const Def* body, const Def* dbg)
        : Def(Node, type, {body}, 0, dbg)
    {}

public:
    /// @name getters
    //@{
    const Def* body() const { return op(0); }
    const Arr* type() const { return Def::type()->as<Arr>(); }
    const Def* shape() const { return type()->shape(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Pack;
    friend class World;
};

inline bool is_sigma_or_arr (const Def* def) { return def->isa<Sigma>() || def->isa<Arr >(); }
inline bool is_tuple_or_pack(const Def* def) { return def->isa<Tuple>() || def->isa<Pack>(); }

/// Extracts from a @p Sigma or @p Variadic typed @p Def the element at position @p index.
class Extract : public Def {
private:
    Extract(const Def* type, const Def* tuple, const Def* index, const Def* dbg)
        : Def(Node, type, {tuple, index}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Extract;
    friend class World;
};

/**
 * Creates a new @p Tuple/@p Pack by inserting @p value at position @p index into @p tuple.
 * @attention { This is a @em functional insert.
 *              The @p tuple itself remains untouched.
 *              The @p Insert itself is a @em new @p Tuple/@p Pack which contains the inserted @p value. }
 */
class Insert : public Def {
private:
    Insert(const Def* tuple, const Def* index, const Def* value, const Def* dbg)
        : Def(Node, tuple->type(), {tuple, index, value}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }
    const Def* value() const { return op(2); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Insert;
    friend class World;
};

/// Flattens a sigma/array/pack/tuple.
const Def* flatten(const Def* def);

/// Applies the reverse transformation on a pack/tuple, given the original type.
const Def* unflatten(const Def* def, const Def* type);
/// Same as unflatten, but uses the operands of a flattened pack/tuple directly.
const Def* unflatten(Defs ops, const Def* type);

Array<const Def*> merge(const Def* def, Defs defs);
const Def* merge_sigma(const Def* def, Defs defs);
const Def* merge_tuple(const Def* def, Defs defs);

bool is_unit(const Def*);
bool is_tuple_arg_of_app(const Def*);

std::string tuple2str(const Def*);

}

#endif
