#ifndef ANYDSL_CFG_H
#define ANYDSL_CFG_H

#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl/symbol.h"
#include "anydsl/util/assert.h"


namespace anydsl {
    class Branch;
    class Def;
    class Lambda;
    class Param;
    class Pi;
    class Type;
    class LVar;
}

namespace anydsl {

class BB;
class Binding;
class Emitter;
class Fct;
class Token;
class World;
typedef boost::unordered_set<BB*> BBs;
typedef std::vector<BB*> BBList;

//------------------------------------------------------------------------------

class BB {
protected:

    BB(BB* parent, const Pi* pi, const std::string& name);
    BB(Fct* fct, World& world, const std::string& name = "");
    virtual ~BB() {}

    static BB* createBB(Fct* fct, World& world, const std::string& name);

public:

    Lambda* lambda() const { return lambda_; }
    std::string name() const;

    void goesto(BB* to);
    void branches(Def* cond, BB* tbb, BB* fbb);
    void invokes(Def* fct);
    void fixto(BB* to);

    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    LVar* getVN(const Symbol sym) { return 0; }
    void setVN(Binding* bind);
    bool hasVN(const Symbol sym) { return values_.find(sym) != values_.end(); }

    //void finalizeAll();
    //void processTodos();
    //void finalize(ParamIter param, const Symbol sym);

    const BBs& pred() const { return pred_; }
    const BBs& succ() const { return succ_; }

protected:

    void dfs(BBList& bbs);

    void flowsto(BB* to);
    World& world();

    typedef boost::unordered_map<Symbol, Binding*> ValueMap;
    ValueMap values_;

    typedef boost::unordered_map<Symbol, size_t> Todos;
    Todos todos_;

    // CFG
    BBs pred_;
    BBs succ_;

    Param* param_;
    Lambda* lambda_;

    //void fixBeta(Beta* beta, size_t x, const Symbol sym, Type* type);

public:

    bool visited_;
    size_t poIndex_; ///< Post-order number -- index to \p Fct::postorder_.

private:

    Fct* fct_; ///< Fct where this BB belongs to.
};

//------------------------------------------------------------------------------

class Fct : public BB {
public:

    Fct(const Pi* pi, const Symbol sym);
    Fct(World& world, const Symbol sym);

    BB* createBB(const std::string& name = "");

    void setReturnCont(const Type* retType);
    bool hasReturnCont() const { return retParam_; }
    void insertReturnStmt(BB* bb, const Def* def);
    virtual Binding* getVN(const Symbol sym, const Type* type, bool finalize);
    BB* exit() const { return exit_; }

    void buildDomTree();
    size_t intersect(size_t i, size_t j);

private:

    BBs cfg_;
    BBList postorder_;
    BBList idoms_;

    BB* exit_;
    Param* retParam_;
};

} // namespace anydsl

#endif
