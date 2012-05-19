#ifndef ANYDSL_SUPPORT_CFG_H
#define ANYDSL_SUPPORT_CFG_H

#include <map>
#include <boost/unordered_set.hpp>

#include "anydsl/util/assert.h"
#include "anydsl/support/symbol.h"


namespace anydsl {
    class Branch;
    class Def;
    class Lambda;
    class Param;
    class Pi;
    class Type;
}

namespace anydsl {

class BB;
class Binding;
class Emitter;
class Fct;
class Token;
class World;
typedef boost::unordered_set<BB*> BBs;
typedef std::list<Param*> Params;
typedef Params::iterator ParamIter;

//------------------------------------------------------------------------------

class BB {
protected:

    BB(BB* parent, const Pi* pi, const std::string& name);
    BB(World& world, const std::string& name = "");
    virtual ~BB() {}

    static BB* createBB(World& world, const std::string& name);

public:

    Lambda* lambda() const { return lambda_; }
    std::string name() const;

    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    void setVN(Binding* bind);
    bool hasVN(const Symbol sym) { return values_.find(sym) != values_.end(); }

    void finalizeAll();
    //void processTodos();
    void finalize(ParamIter param, const Symbol sym);

    const BBs& pred() const { return pred_; }
    const BBs& succ() const { return succ_; }
    void flowsto(BB* to);

protected:

    /// Insert \p bb as sub BB (i.e., as dom child) into this BB.
    void insert(BB* bb);
    World& world();

    typedef std::map<const Symbol, Binding*> ValueMap;
    ValueMap values_;

    typedef std::map<Symbol, ParamIter, Symbol::FastLess> Todos;
    Todos todos_;

    // dominator tree
    BB* parent_;
    BBs children_;

    // CFG
    BBs pred_;
    BBs succ_;

    Param* param_;
    Lambda* lambda_;

    //void fixBeta(Beta* beta, size_t x, const Symbol sym, Type* type);

    bool finalized_;
};

//------------------------------------------------------------------------------

class Fct : public BB {
public:

    Fct(const Pi* pi, const Symbol sym);
    Fct(World& world, const Symbol sym);

    BB* createBB(const std::string& name = "");

    void goesto(BB* from, BB* to);
    void branches(BB* from, Def* cond, BB* tbb, BB* fbb);
    void invokes(BB* from, Def* fct);
    void fixto(BB* from, BB* to);

    void setReturn(const Type* retType);
    bool hasReturn() const { return ret_; }
    void insertReturn(BB* bb, Def* def);
    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    BB* exit() const { return exit_; }

private:

    BBs cfg_;

    BB* exit_;
    Param* ret_;
};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_CFG_H
