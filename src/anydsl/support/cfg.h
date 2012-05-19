#ifndef ANYDSL_SUPPORT_CFG_H
#define ANYDSL_SUPPORT_CFG_H

#include <map>
#include <boost/unordered_set.hpp>

#include "anydsl/util/assert.h"
#include "anydsl/support/symbol.h"
#include "impala/value.h"


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
typedef std::list<const Param*> Params;
typedef Params::iterator ParamIter;

//------------------------------------------------------------------------------

class BB {
protected:

    BB(BB* parent, const Pi* pi, const std::string& name = "");

public:

    BB(World& world, const std::string& name = "");
    virtual ~BB() {}

    /// Insert \p bb as sub BB (i.e., as dom child) into this BB.
    void insert(BB* bb);

    void goesto(BB* to);
    void branches(Def* cond, BB* tbb, BB* fbb);
    void invokes(Def* fct);
    void fixto(BB* to);

    const BBs& pre() const { return pred_; }
    const BBs& succ() const { return succ_; }
    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    void setVN(Binding* bind);

    void finalizeAll();
    //void processTodos();
    void finalize(ParamIter param, const Symbol sym);
    //bool hasVN(const Symbol sym) { return values_.find(sym) != values_.end(); }

    Lambda* lambda() const { return lambda_; }
    std::string name() const;

#ifndef NDEBUG
    Symbol belongsTo();
#endif

    World& world();

protected:
public:

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

#ifndef NDEBUG
    bool verify(BB* bb);
#endif

private:

    //void fixBeta(Beta* beta, size_t x, const Symbol sym, Type* type);

    bool finalized_;

    //friend class impala::Emitter;
};

//------------------------------------------------------------------------------

class Fct : public BB {
public:

    Fct(BB* parent, const Pi* pi, const Symbol sym);
    Fct* createSubFct(const Pi* pi, const Symbol sym);

    void setReturn(const Location& loc, Type* retType);
    bool hasReturn() const { return ret_; }
    void insertReturn(const Location& loc, BB* bb, Def* def);
    void insertCont(const Location& loc, BB* where, Def* cont);
    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);

private:

    BB* exit_;
    Param* ret_;

    //friend class impala::Emitter;
};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_CFG_H
