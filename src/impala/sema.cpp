#include "impala/ast.h"

#include "impala/dump.h"

#include <stack>

#include "anydsl/util/for_all.h"

using anydsl::Location;
using anydsl::Symbol;

namespace impala {

//------------------------------------------------------------------------------

class Sema {
public:

    Sema();
    ~Sema();

    /** 
     * @brief Looks up the current definition of \p sym.
     * 
     * @param sym The \p Symbol to look up.
     * 
     * @return Returns 0 on failure.
     */
    const Decl* lookup(const anydsl::Symbol sym);

    /** 
     * @brief Pushes a new Decl on the internal \p DeclStack.
     * 
     * If sym already has a definition in this scope an assertion is raised.
     * use \p clash in order to check this.
     *
     * @param sym The \p Symbol to insert.
     */
    void insert(const Decl* decl);

    /** 
     * @brief Checks whether there already exists a \p Symbol \p sym in the \em current scope.
     * 
     * @param sym The \p Symbol to check.
     * 
     * @return The current mapping if the lookup succeeds, 0 otherwise.
     */
    const Decl* clash(const anydsl::Symbol sym) const;

    /// Open a new scope.
    void pushScope();
    /// Discard current scope.
    void popScope();

    bool result() const { return result_; }

    std::ostream& error(const ASTNode* n) { result_ = false; return n->emitError(); }
    std::ostream& warning(const ASTNode* n) { return n->emitWarning(); }

private:

    struct Slot {
        Slot(const Decl* decl, int depth)
            : decl(decl)
            , depth(depth)
        {}
        Slot() {}

        const Decl* decl;
        int depth;
    };

    typedef std::stack<Slot> SlotStack;
    typedef std::map<const anydsl::Symbol, SlotStack*, anydsl::Symbol::FastLess> Scope;

    bool result_;
    Scope scope_;
    int depth_;

#ifndef NDEBUG
    int refcounter_;
#endif
};

//------------------------------------------------------------------------------

Sema::Sema()
    : result_(true)
    , depth_(0)
#ifndef NDEBUG
    , refcounter_(0)
#endif
{
    pushScope();
}

Sema::~Sema() {
    anydsl_assert(depth_ == 1, "root scope must be 1");
    popScope();

#ifndef NDEBUG
    anydsl_assert(refcounter_ == 0, "memory leak");
#endif
}

const Decl* Sema::lookup(const Symbol sym) {
    Scope::iterator i = scope_.find(sym);
    if (i != scope_.end())
        return i->second->top().decl;
    else
        return 0;
}

void Sema::insert(const Decl* decl) {
    const Symbol sym = decl->symbol();

    anydsl_assert(clash(sym) == 0, "must not be found");

    // create stack if necessary
    Scope::iterator i = scope_.find(sym);
    if (i == scope_.end()) {
        // isn't C++ beautiful? We should switch to C++11 just because of auto
        std::pair<Scope::iterator, bool> p =
            scope_.insert(std::make_pair(sym, new SlotStack()));
#ifndef NDEBUG
        ++refcounter_;
#endif
        i = p.first;
    }

    // get pointer to stack linked to sym and push new Decl
    SlotStack* stack = i->second;
    stack->push(Slot(decl, depth_));
}

const Decl* Sema::clash(const Symbol sym) const {
    Scope::const_iterator i = scope_.find(sym);
    if (i != scope_.end() && i->second->top().depth == depth_)
        return i->second->top().decl;
    else
        return 0;
}

void Sema::popScope() {
    anydsl_assert(depth_ > 0, "illegal depth value");

    for_all (i, scope_) {
        const Symbol& sym = i.first;
        SlotStack* stack = i.second;
        anydsl_assert(!stack->empty(), "must have at least on element");
        
        if (depth_ == stack->top().depth) {
            stack->pop();

            if (stack->empty()) {
                scope_.erase(sym);
                delete stack;
#ifndef NDEBUG
                --refcounter_;
#endif
            }
        }
    }

    --depth_;
}

void Sema::pushScope() {
    ++depth_;
}

//------------------------------------------------------------------------------

bool check(const Prg* prg) {
    Sema sema;
    prg->check(sema);
    return sema.result();
}

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) const {
    for_all (f, fcts())
        f->check(sema);
}

void Fct::check(Sema& sema) const {
    sema.pushScope();

    for_all (p, params())
        p->check(sema);

    if (retType())
        retType()->check(sema);

    for_all (const &s, body()->stmts())
        s->check(sema);

    sema.popScope();
}

void Decl::check(Sema& sema) const {
    type()->check(sema);

    if (const Decl* decl = sema.clash(symbol())) {
        sema.error(this) << "symbol '" << symbol() << "' already defined\n";
        sema.error(decl) << "previous location here\n";
    } else {
        sema.insert(this);
    }
}

/*
 * Type
 */

void PrimType::check(Sema& sema) const {
    /* do nothing */
}

void Void::check(Sema& sema) const {
    /* do nothing */
}

void ErrorType::check(Sema& sema) const {
    /* do nothing */
}


/*
 * Expr
 */

void EmptyExpr::check(Sema& sema) const {
    type_ = new Void(loc());
    lvalue_ = false;
}

void Literal::check(Sema& sema) const {
    PrimType::Kind newKind;

    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case Literal::LIT_##itype: \
            newKind = PrimType::TYPE_##itype; \
            break;
#include "impala/tokenlist.h"
        case Literal::LIT_bool:
            newKind = PrimType::TYPE_bool;
            break;
    }

    type_ = new PrimType(loc(), newKind);
    lvalue_ = false;
}

void Id::check(Sema& sema) const {
    lvalue_ = true;

    if (const Decl* decl = sema.lookup(symbol())) {
        type_ = decl->type()->clone(loc());
        return;
    }

    sema.error(this) << "symbol '" << symbol() << "' not found in current scope\n";
    type_ = new ErrorType(loc());
}

void PrefixExpr::check(Sema& sema) const {
    rexpr()->check(sema);
    type_ = rexpr()->type()->clone(loc());
    lvalue_ = true;
}

void InfixExpr::check(Sema& sema) const {
    lexpr()->check(sema);
    rexpr()->check(sema);

    Location loc(lexpr()->pos1(), rexpr()->pos2());

    bool equal = lexpr()->type()->equal(rexpr()->type());

    if (!equal) {
        sema.error(this) << "incompatible types in binary expression: '" 
            << lexpr()->type() << "' and '" << rexpr()->type() << "'\n";
    }

    if (Token::isRel((TokenKind) kind())) {
        type_ = new PrimType(loc, PrimType::TYPE_bool);
        lvalue_ = false;
        return;
    }

    if (Token::isAsgn((TokenKind) kind())) {
        if (!lexpr()->lvalue())
            sema.error(lexpr()) << "no lvalue on left-hand side of assignment\n";

        lvalue_ = true;
    }

    if (!lexpr()->type()->isError())
        type_ = lexpr()->type()->clone(loc);
    else
        type_ = rexpr()->type()->clone(loc);
}

void PostfixExpr::check(Sema& sema) const {
    lvalue_ = false;

    lexpr()->check(sema);
    type_ = lexpr()->type()->clone(loc());
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const {
    decl()->check(sema);
}

void ExprStmt::check(Sema& sema) const {
    expr()->check(sema);
}

static bool checkCond(Sema& sema, const Expr* cond) {
    cond->check(sema);

    if (!cond->type()->isBool()) {
        sema.error(cond) << "condition not a bool\n";
        return false;
    }

    return true;
}

void IfElseStmt::check(Sema& sema) const {
    checkCond(sema, cond());
    ifStmt()->check(sema);
    elseStmt()->check(sema);
}

void WhileStmt::check(Sema& sema) const {
    checkCond(sema, cond());
    body()->check(sema);
}

void DoWhileStmt::check(Sema& sema) const {
    body()->check(sema);
    checkCond(sema, cond());
}

void ForStmt::check(Sema& sema) const {
    init()->check(sema);
    checkCond(sema, cond());
    inc()->check(sema);
    body()->check(sema);
}

void BreakStmt::check(Sema& sema) const {
    if (!loop())
        sema.error(this) << "break statement not within a loop\n";
}

void ContinueStmt::check(Sema& sema) const {
    if (!loop())
        sema.error(this) << "continue statement not within a loop\n";
}

void ReturnStmt::check(Sema& sema) const {
    expr()->check(sema);

    if (!fct()->retType()->equal(expr()->type())) {
        sema.error(expr()) << "expected return type '" << fct()->retType() 
            << "' but return expression is of type '" << expr()->type() << "'\n";
    }
}

void ScopeStmt::check(Sema& sema) const {
    sema.pushScope();

    for_all (const &s, stmts())
        s->check(sema);

    sema.popScope();
}

} // namespace impala
