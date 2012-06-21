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
class LVar;
class Lambda;
class Pi;
class World;

typedef std::vector<const Def*> Defs;
typedef boost::unordered_set<BB*> BBs;

class BB {
private:

    BB(Fct* fct, const std::string& debug = "");

public:

    void setVar(const LVar& lvar);
    LVar getVar(const Symbol& symbol);
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

    typedef boost::unordered_map<Symbol, LVar> ValueMap;
    ValueMap values_;

    typedef boost::unordered_map<Symbol, size_t> Todos;
    Todos todos_;

    friend class Fct;
};

class Fct {
public:

    Fct(const Symbol& symbol, const Pi* pi);

    BB* createBB(const std::string& debug = "");
    const Pi* pi() const { return pi_; }

private:

    const Pi* pi_;
    const Lambda* lambda_;
    BBs cfg_;
};

} // namespace anydsl

#endif
