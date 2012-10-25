#ifndef ANYDSL2_CFG_H
#define ANYDSL2_CFG_H

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl2/symbol.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/types.h"

namespace anydsl2 {

class BB;
class Def;
class Fct;
class Lambda;
class Param;
class Pi;
class Type;
class World;

typedef std::vector<const Param*> In;
typedef std::vector<const Def*> Out;
typedef boost::unordered_set<BB*> BBs;
typedef boost::unordered_map<Symbol, Fct*> LetRec;

class Todo {
public:

    Todo() {}
    Todo(size_t index, const Type* type)
        : index_(index)
        , type_(type)
    {}

    size_t index() const { return index_; }
    const Type* type() const { return type_; }

private:

    size_t index_;
    const Type* type_;
};


class Var {
public:

    Var() {}
    Var(Symbol symbol, const Def* def)
        : symbol_(symbol)
        , def_(def)
    {}
    virtual ~Var() {}

    Symbol symobl() const { return symbol_; }
    virtual const Def* load() const { return def_; }
    virtual void store(const Def* def) { def_ = def; }

protected:

    Symbol symbol_;
    const Def* def_;
};

/** 
 * This class helps for code generation of imperative languages.
 *
 * SSA/CPS construction is supported via \p lookup and \p insert.
 * In order to make this work a \p BB must be aware of the fact whether all predecessors are known
 * or whether there may still be predecessors added.
 * A \em sealed \p BB knows all its predecessors.
 * It is prohibited to add additional predecessors on a sealed \p BB.
 * The construction algorithm works best, if you \p seal a \p BB as soon as possible, i.e., 
 * as soon as you know that a \p BB cannot get any more predecessors invoke \p seal.
 */
class BB {
private:

    BB(Fct* fct, const std::string& debug = "");
    BB() {}
    ~BB();

public:

    Var* insert(Symbol symbol, const Def* def);
    Var* lookup(Symbol symbol, const Type* type);
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
    const Lambda* top() const { return top_; }
    const Lambda* cur() const { return cur_; }

    World& world();
    bool sealed() const { return sealed_; }
    std::string debug() const;

    void emit();

private:

    void link(BB* to);
    void fix(Symbol symbol, Todo todo);

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

    typedef boost::unordered_map<Symbol, Var*> VarMap;
    VarMap vars_;

    typedef boost::unordered_map<Symbol, Todo> Todos;
    Todos todos_;

    friend class Fct;
};

class Fct : public BB {
public:

    Fct(World& world)
        : world_(world) 
    {
        fct_ = this;
    }
    Fct(World& world, const Pi* pi, ArrayRef<Symbol> symbols, 
        size_t return_index, const std::string& debug);
    ~Fct();

    BB* createBB(const std::string& debug = "");
    void emit();
    World& world() { return world_; }
    Var* lookup_top(Symbol symbol, const Type* type);
    void nest(Symbol symbol, Fct* fct);
    const Param* ret() const { return ret_; }

    BB* parent() const { return parent_; }
    void set_parent(BB* parent) { parent_ = parent; }
    LetRec& letrec() { return letrec_; }

private:

    World& world_;
    const Param* ret_;
    BB* parent_;
    std::vector<BB*> cfg_;
    LetRec letrec_;
};

} // namespace anydsl2

#endif
