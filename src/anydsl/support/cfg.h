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

    BB(BB* parent, const Pi* pi, const std::string& name = "");
    BB(World& world, const std::string& name = "");
    virtual ~BB() {}

public:

    Lambda* lambda() const { return lambda_; }
    std::string name() const;

    void goesto(BB* to);
    void branches(Def* cond, BB* tbb, BB* fbb);
    void invokes(Def* fct);
    void fixto(BB* to);

    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    void setVN(Binding* bind);
    bool hasVN(const Symbol sym) { return values_.find(sym) != values_.end(); }

    void finalizeAll();
    //void processTodos();
    void finalize(ParamIter param, const Symbol sym);

protected:

    /// Insert \p bb as sub BB (i.e., as dom child) into this BB.
    void insert(BB* bb);
    World& world();

    typedef std::map<const Symbol, Binding*> ValueMap;
    ValueMap values_;

    typedef std::map<Symbol, ParamIter, Symbol::FastLess> Todos;
    Todos todos_;

    void flowsTo(BB* to);

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

    //friend class Fct;
};

//------------------------------------------------------------------------------

class Fct : public BB {
private:

    Fct(Fct* parent, const Pi* pi, const Symbol sym);

public:

    /**
     * Use this constructor to create the root function 
     * which holds the whole program.
     */
    Fct(World& world, const Symbol sym = "<root-function>");

    Fct* createSubFct(const Pi* pi, const Symbol sym);

    void setReturn(const Type* retType);
    bool hasReturn() const { return ret_; }
    void insertReturn(BB* bb, Def* def);
    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);

private:

    BBs cfg_;

    BB* exit_;
    Param* ret_;
};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_CFG_H
