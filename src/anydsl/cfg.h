#ifndef ANYDSL_CFG_H
#define ANYDSL_CFG_H

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl/symbol.h"
#include "anydsl/util/array.h"

namespace anydsl {

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
    Var(const Symbol& symbol, const Def* def)
        : symbol_(symbol)
        , def_(def)
    {}

    Symbol symobl() const { return symbol_; }
    const Def* load() const { return def_; }
    void store(const Def* def) { def_ = def; }

private:

    Symbol symbol_;
    const Def* def_;
};

/** 
 * This class helps for code generation of imperative languages.
 *
 * SSA/CPS construction is supported via \p getVar and \p setVar.
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

    Var* setVar(const Symbol& symbol, const Def* def);
    Var* getVar(const Symbol& symbol, const Type* type);
    void seal();

    void goesto(BB* to);
    void branches(const Def* cond, BB* tbb, BB* fbb);
    void fixto(BB* to);
    const Def* calls(const Def* to, ArrayRef<const Def*> args, const Type* retType);

    const BBs& preds() const { return preds_; }
    const BBs& succs() const { return succs_; }

    const Lambda* topLambda() const { return topLambda_; }
    const Lambda* curLambda() const { return curLambda_; }

    World& world();
    bool sealed() const { return sealed_; }

    void emit();

private:

    void flowsto(BB* to);
    void fixTodo(const Symbol& symbol, Todo todo);

    bool sealed_;

    Fct* fct_;

    In in_;
    Out out_;

    BBs preds_;
    BBs succs_;
    const Def* cond_;
    BB* tbb_;
    BB* fbb_;

    Lambda* topLambda_;
    Lambda* curLambda_;

    typedef boost::unordered_map<Symbol, Var*> VarMap;
    VarMap vars_;

    typedef boost::unordered_map<Symbol, Todo> Todos;
    Todos todos_;

    friend class Fct;
};

class Fct : public BB {
public:

    Fct(World& world, 
        ArrayRef<const Type*> tparams, ArrayRef<Symbol> sparams, 
        const Type* retType, const std::string& debug = "");
    ~Fct();

    BB* createBB(const std::string& debug = "");
    void emit();
    World& world() { return world_; }

    BB* exit() { return exit_; }
    const Param* retCont() { return retCont_; }
    const Type* retType() { return retType_; }

private:

    World& world_;
    const Type* retType_;
    const Param* retCont_;
    BB* exit_;
    BBs cfg_;
};

} // namespace anydsl

#endif
