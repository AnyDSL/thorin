#ifndef ANYDSL_SUPPORT_CFG_H
#define ANYDSL_SUPPORTDING_H

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
    class Type;
}

namespace anydsl {

class BB;
class Emitter;
class Fct;
class Token;
class World;
typedef boost::unordered_set<BB*> BBs;

//------------------------------------------------------------------------------

class BB {
public:

    BB(World& world, const std::string& name = "");
    virtual ~BB() {}

    //static BB* create(const Symbol sym = Symbol(""));

    /// Insert \p bb as sub BB (i.e., as dom child) into this BB.
    void insert(BB* bb);

    void goesto(BB* to);
    void branches(Def* cond, BB* tbb, BB* fbb);
    void invokes(Def* fct);

    const BBs& pre() const { return pred_; }
    const BBs& succ() const { return succ_; }
    //Def* appendLambda(CExpr* cexpr, Type* type);
    //virtual Binding* getVN(const Location& loc, const Symbol sym, Type* type, bool finalize);
    //void setVN(const Location& loc, Binding* bind);

    void finalizeAll();
    //void processTodos();
    //void finalize(size_t x, const Symbol sym);
    //bool hasVN(const Symbol sym) { return values_.find(sym) != values_.end(); }

    void inheritValues(BB* bb);

    Lambda* lambda() const { return lambda_; }
    std::string name() const;

#ifndef NDEBUG
    Symbol belongsTo();
#endif

    World& world();

protected:
public:

#if 0

    typedef std::map<const Symbol, Binding*> ValueMap;
    ValueMap values_;

    typedef std::map<Symbol, int, Symbol::FastLess> Todos;
    Todos todos_;
#endif

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

#if 0
class Fct : public BB {
public:

    Fct(const Location& loc, const Symbol sym);
    Fct(BB* parent, const Location& loc, const Symbol sym);

    //Fct* createSubFct(const Location& loc, const Symbol sym);
    void setReturn(const Location& loc, Type* retType);
    bool hasReturn() const { return ret_; }
    void insertReturn(const Location& loc, BB* bb, Def* def);
    void insertCont(const Location& loc, BB* where, Def* cont);
    //Def* appendLambda(BB* bb, CExpr* cexpr, Type* type);
    //virtual Binding* getVN(const Location& loc, const Symbol, Type* type, bool finalize);

private:

    BB* exit_;
    Param* ret_;

    //friend class impala::Emitter;
};
#endif

} // namespace anydsl

#endif // ANYDSL_SUPPORT_CFG_H
