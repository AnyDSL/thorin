#ifndef ANYDSL_CFG_H
#define ANYDSL_CFG_H

#include <string>
#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl/symbol.h"

namespace anydsl {

class BB;
class Def;
class Fct;
class Lambda;
class Pi;
class Type;
class World;

typedef std::vector<const Def*> Defs;
typedef boost::unordered_set<BB*> BBs;

struct FctParam {
    Symbol symbol;
    const Type* type;

    FctParam() {}
    FctParam(const Symbol& symbol, const Type* type)
        : symbol(symbol)
        , type(type)
    {}
};

struct Todo {
    size_t index;
    const Type* type;

    Todo() {}
    Todo(size_t index, const Type* type)
        : index(index)
        , type(type)
    {}
};

typedef std::vector<FctParam> FctParams;

struct Var {
    Symbol symbol;
    const Def* def;

    Var() {}
    Var(const Symbol& symbol, const Def* def)
        : symbol(symbol)
        , def(def)
    {}
};

class BB {
private:

    BB(Fct* fct, const std::string& debug = "");
    BB() {}

public:

    Var* setVar(const Symbol& symbol, const Def* def);
    Var* getVar(const Symbol& symbol, const Type* type);
    void seal();

    void goesto(BB* to);
    void branches(const Def* cond, BB* tbb, BB* fbb);
    void fixto(BB* to);

    const BBs& preds() const { return preds_; }
    const BBs& succs() const { return succs_; }

    const Lambda* topLambda() const { return topLambda_; }

    World& world();
    bool sealed() const { return sealed_; }

private:

    void flowsto(BB* to);

    bool sealed_;

    Fct* fct_;

    Defs in_;
    Defs out_;

    const Def* to_;

    BBs preds_;
    BBs succs_;

    Lambda* topLambda_;
    Lambda* curLambda_;

    typedef boost::unordered_map<Symbol, Var*> ValueMap;
    ValueMap values_;

    typedef boost::unordered_map<Symbol, Todo> Todos;
    Todos todos_;

    friend class Fct;
};

class Fct : public BB {
public:

    Fct(const FctParams& fparams, const Type* retType, const std::string& debug = "");

    BB* createBB(const std::string& debug = "");

private:

    BBs cfg_;
    const Type* retType_;
};

} // namespace anydsl

#endif
