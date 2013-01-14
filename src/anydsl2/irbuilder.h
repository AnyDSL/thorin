#ifndef ANYDSL2_IRBUILDER_H
#define ANYDSL2_IRBUILDER_H

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl2/symbol.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/types.h"
#include "anydsl2/util/indexmap.h"

namespace anydsl2 {

class BB;
class Def;
class Fct;
class Generic;
class Lambda;
class Param;
class Pi;
class Type;
class World;

typedef std::vector<const Param*> In;
typedef std::vector<const Def*> Out;
typedef boost::unordered_set<BB*> BBs;

//------------------------------------------------------------------------------

class Todo {
public:

    Todo() {}
    Todo(size_t handle, size_t index, const Type* type)
        : handle_(handle)
        , index_(index)
        , type_(type)
    {}

    size_t handle() const { return handle_; }
    size_t index() const { return index_; }
    const Type* type() const { return type_; }

private:

    size_t handle_;
    size_t index_;
    const Type* type_;
};

//------------------------------------------------------------------------------

/** 
 * This class helps for code generation of imperative languages.
 *
 * SSA/CPS construction is supported via \p get_value and \p set_value.
 * In order to make this work a \p BB must be aware of the fact whether all predecessors are known
 * or whether there may still be predecessors added.
 * A \em sealed \p BB knows all its predecessors.
 * It is prohibited to add additional predecessors on a sealed \p BB.
 * The construction algorithm works best, if you \p seal a \p BB as soon as possible, i.e., 
 * as soon as you know that a \p BB cannot get any more predecessors invoke \p seal.
 */
class BB {
private:

    BB(Fct* fct, const std::string& name = "");
    BB() {}

public:

    void set_value(size_t handle, const Def* def);
    const Def* get_value(size_t handle, const Type* type, const std::string& name = "");
    void seal();

    void jump(BB* to);
    void branch(const Def* cond, BB* tbb, BB* fbb);
    void fixto(BB* to);
    const Def* call(const Def* to, ArrayRef<const Def*> args, const Type* rettype);
    void tail_call(const Def* to, ArrayRef<const Def*> args);
    void return_tail_call(const Def* to, ArrayRef<const Def*> args);
    void return_void();
    void return_value(const Def* result);

    const BBs& preds() const { return preds_; }
    const BBs& succs() const { return succs_; }
    Lambda* top() const { return top_; }
    Lambda* cur() const { return cur_; }

    World& world();
    bool sealed() const { return sealed_; }
    std::string name() const;

    void emit();

private:

    void link(BB* to);
    void fix(Todo todo);

    bool sealed_;
    bool visited_;

    Fct* fct_;

    In in_;
    Out out_;

    BBs preds_;
    BBs succs_;
    const Def* cond_;
    BB* tbb_;
    BB* fbb_;

    Lambda* top_;
    Lambda* cur_;

    typedef IndexMap<const Def> DefMap;
    DefMap defs_;

    typedef std::vector<Todo> Todos;
    Todos todos_;

    friend class Fct;
};

//------------------------------------------------------------------------------

class Fct : public BB {
public:

    Fct(World& world)
        : world_(world) 
    {
        fct_ = this;
    }
    Fct(World& world, const Pi* pi, ArrayRef<size_t> handles, ArrayRef<Symbol> symbols, 
        size_t return_index, const std::string& name);
    ~Fct();

    BB* createBB(const std::string& name = "");
    void emit();
    World& world() { return world_; }
    const Def* get_value_top(size_t handle, const Type* type, const std::string& name);
    const Param* ret() const { return ret_; }

    BB* parent() const { return parent_; }
    void set_parent(BB* parent) { parent_ = parent; }

private:

    World& world_;
    const Param* ret_;
    BB* parent_;
    std::vector<BB*> cfg_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
