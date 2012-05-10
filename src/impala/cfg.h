#ifndef IMPALA_CFG_H
#define CF_BINDING_H

#include <map>

#include "anydsl/util/assert.h"
#include "anydsl/support/symbol.h"
#include "impala/value.h"


#if 0

namespace anydsl {
    class Beta;
    class Branch;
    class CExpr;
    class Def;
    class Fix;
    class Lambda;
    class Location;
    class Param;
    class Type;
}

namespace impala {

class BB;
class Emitter;
class Fct;
class Parser;
class Token;
typedef anydsl::Set<BB*> BBList;

//------------------------------------------------------------------------------

class BB {
public:

    BB(BB* parent, const anydsl::Location& loc, const anydsl::Symbol sym = anydsl::Symbol(""));
    virtual ~BB() {}

    static BB* create(const anydsl::Symbol sym = anydsl::Symbol(""));

    BB* createSubBB(const anydsl::Location& loc, const anydsl::Symbol sym = anydsl::Symbol(""));
    void hangInBB(BB* root);
    void jumps(const anydsl::Location& loc, BB* to);
    void calls(const anydsl::Location& loc, anydsl::Def* f);
    void branches(const anydsl::Location& loc, anydsl::Def* cond, BB* toT, BB* toF);
    void mergeChild();

    const BBList& pre() const { return pred_; }
    const BBList& succ() const { return succ_; }
    anydsl::Def* appendLambda(anydsl::CExpr* cexpr, anydsl::Type* type);
    virtual Binding* getVN(const anydsl::Location& loc, const anydsl::Symbol sym, anydsl::Type* type, bool finalize);
    void setVN(const anydsl::Location& loc, Binding* bind);

    void finalizeAll();
    void processTodos();
    void finalize(size_t x, const anydsl::Symbol sym);
    bool hasVN(const anydsl::Symbol sym) { return values_.find(sym) != values_.end(); }

    void inheritValues(BB* bb);

    const char* name();

#ifndef NDEBUG
    anydsl::Symbol belongsTo();
#endif

protected:
public:

    typedef std::map<const anydsl::Symbol, Binding*> ValueMap;
    ValueMap values_;

    typedef std::map<anydsl::Symbol, int, anydsl::Symbol::FastLess> Todos;
    Todos todos_;

    void flowsTo(BB* to);

    // dominator tree
    BB* parent_;
    BBList children_;

    // CFG
    BBList pred_;
    BBList succ_;

    anydsl::Param* param_;
    anydsl::Lambda* lambda_;
    anydsl::Fix* fix_;

#ifndef NDEBUG
    bool verify(BB* bb);
#endif


    anydsl::Beta* getBeta();
    anydsl::Branch* getBranch();

private:

    void fixBeta(anydsl::Beta* beta, size_t x, const anydsl::Symbol sym, anydsl::Type* type);

    /// keeps track of the last thing in this BB
    union {
        anydsl::Lambda* lcursor_;///< points to the lambda in an unfinished BB
        anydsl::Beta* beta_;     ///< points to the jump in a finished BB with one successor
        anydsl::Branch* branch_; ///< points to the branch in a finished BB with two successors
    };

    bool finalized_;

    bool hasTerminator_;
    void setTerminator() { anydsl_assert(!hasTerminator_, "already set"); hasTerminator_ = true; }
    bool hasTerminator() { return hasTerminator_; }


    friend class impala::Emitter;
    friend class impala::Parser;
};

//------------------------------------------------------------------------------

class Fct : public BB {
public:

    Fct(const anydsl::Location& loc, const anydsl::Symbol sym);
    Fct(BB* parent, const anydsl::Location& loc, const anydsl::Symbol sym);

    Fct* createSubFct(const anydsl::Location& loc, const anydsl::Symbol sym);
    void setReturn(const anydsl::Location& loc, anydsl::Type* retType);
    bool hasReturn() const { return ret_; }
    void insertReturn(const anydsl::Location& loc, BB* bb, anydsl::Def* def);
    void insertCont(const anydsl::Location& loc, BB* where, anydsl::Def* cont);
    anydsl::Def* appendLambda(BB* bb, anydsl::CExpr* cexpr, anydsl::Type* type);
    virtual Binding* getVN(const anydsl::Location& loc, const anydsl::Symbol, anydsl::Type* type, bool finalize);

private:

    BB* exit_;
    anydsl::Param* ret_;

    friend class impala::Emitter;
    friend class impala::Parser;
};

} // namespace impala

#endif

#endif // IMPALA_CFG_H
