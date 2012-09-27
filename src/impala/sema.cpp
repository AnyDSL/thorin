#include "impala/ast.h"

#include <stack>
#include <boost/unordered_map.hpp>

#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

#include "impala/dump.h"
#include "impala/type.h"

using anydsl::Array;
using anydsl::Location;
using anydsl::Symbol;

namespace impala {

//------------------------------------------------------------------------------

class Sema {
public:

    Sema(TypeTable& typetable);
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
     * @param decl The \p decl to insert.
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
    void pushScope() { ++depth_; }
    /// Discard current scope.
    void popScope();

    bool result() const { return result_; }

    std::ostream& error(const ASTNode* n) { result_ = false; return n->error(); }
    std::ostream& warning(const ASTNode* n) { return n->warning(); }

    TypeTable& types;

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
    typedef boost::unordered_map<const anydsl::Symbol, SlotStack*> Scope;

    bool result_;
    Scope scope_;
    int depth_;

#ifndef NDEBUG
    int refcounter_;
#endif
};

//------------------------------------------------------------------------------

Sema::Sema(TypeTable& types)
    : types(types)
    , result_(true)
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

    Scope::iterator i = scope_.begin(); 
    while (i != scope_.end()) {
        SlotStack* stack = i->second;
        anydsl_assert(!stack->empty(), "must have at least on element");
        
        if (depth_ == stack->top().depth) {
            stack->pop();

            if (stack->empty()) {
                i = scope_.erase(i);
                delete stack;
#ifndef NDEBUG
                --refcounter_;
#endif
                continue;
            }
        }

         ++i;
    }

    --depth_;
}

//------------------------------------------------------------------------------

bool check(TypeTable& types, const Prg* prg) {
    Sema sema(types);
    prg->check(sema);
    return sema.result();
}

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) const {
    for_all (f, fcts())
        sema.insert(f->decl());

    for_all (f, fcts())
        f->check(sema);
}

void Fct::check(Sema& sema) const {
    sema.pushScope();

    for_all (p, params())
        p->check(sema);

    for_all (s, body()->stmts())
        s->check(sema);

    sema.popScope();
}

void Decl::check(Sema& sema) const {
    if (const Decl* decl = sema.clash(symbol())) {
        sema.error(this) << "symbol '" << symbol() << "' already defined\n";
        sema.error(decl) << "previous location here\n";
    } else
        sema.insert(this);
}

/*
 * Expr
 */

const Type* EmptyExpr::vcheck(Sema& sema) const {
    return sema.types.type_void();
}

const Type* Literal::vcheck(Sema& sema) const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case Literal::LIT_##itype:  return sema.types.type(PrimType::TYPE_##itype);
#include "impala/tokenlist.h"
        case Literal::LIT_bool:     return sema.types.type(PrimType::TYPE_bool);
        default:                    ANYDSL_UNREACHABLE;
    }
}

const Type* Tuple::vcheck(Sema& sema) const {
    Array<const Type*> elems(ops().size());

    size_t i = 0;
    for_all (op, ops())
        elems[i++] = op->check(sema);

    return sema.types.sigma(elems);
}

const Type* Id::vcheck(Sema& sema) const {
    if (const Decl* decl = sema.lookup(symbol())) {
        decl_ = decl;
        return decl->type();
    }

    sema.error(this) << "symbol '" << symbol() << "' not found in current scope\n";
    return sema.types.type_error();
}

const Type* PrefixExpr::vcheck(Sema& sema) const {
    if (!rhs()->lvalue())
        sema.error(rhs()) << "lvalue required as operand\n";

    return rhs()->check(sema);
}

const Type* InfixExpr::vcheck(Sema& sema) const {
    if (lhs()->check(sema) == rhs()->check(sema)) {
        if (Token::is_rel((TokenKind) kind()))
            return sema.types.type_bool();

        if (Token::is_asgn((TokenKind) kind())) {
            if (!lhs()->lvalue())
                sema.error(lhs()) << "no lvalue on left-hand side of assignment\n";
        }

        if (lhs()->type()->is_error())
            return rhs()->type();
        else
            return lhs()->type();
    } else {
        sema.error(this) << "incompatible types in binary expression: '" 
            << lhs()->type() << "' and '" << rhs()->type() << "'\n";
    }

    return sema.types.type_error();
}

const Type* PostfixExpr::vcheck(Sema& sema) const {
    if (!lhs()->lvalue())
        sema.error(lhs()) << "lvalue required as operand\n";

    return lhs()->check(sema);
}

const Type* IndexExpr::vcheck(Sema& sema) const {
    if (const Sigma* sigma = lhs()->check(sema)->isa<Sigma>()) {
        if (index()->check(sema)->is_int()) {
            if (const Literal* literal = index()->isa<Literal>()) {
                unsigned pos;

                switch (literal->kind()) {
#define IMPALA_LIT(itype, atype) \
                    case Literal::LIT_##itype: pos = literal->box().get_##atype(); break;
#include "impala/tokenlist.h"
                    default: ANYDSL_UNREACHABLE;
                }

                if (pos < sigma->size())
                    return sigma->elems()[pos];
                else
                    sema.error(index()) << "index (" << pos << ") out of bounds (" << sigma->size() << ")\n";
            } else
                sema.error(index()) << "indexing expression must be a literal\n";
        } else
            sema.error(index()) << "indexing expression must be of integer type\n";
    } else
        sema.error(lhs()) << "left-hand side of index expression must be of sigma type\n";

    return sema.types.type_error();
}

const Type* Call::vcheck(Sema& sema) const { 
    if (const Pi* fpi = to()->check(sema)->isa<Pi>()) {
        Array<const Type*> op_types(ops_.size() - 1);

        for (size_t i = 1; i < ops_.size(); ++i)
            op_types[i-1] = ops_[i]->check(sema);

        const Pi* pi = sema.types.pi(op_types, fpi->ret());

        if (pi == fpi)
            return pi->ret();
        else {
            sema.error(to()) << "'" << to() << "' expects an invocation of type '" << fpi 
                << "' but an invocation of type '" << pi << "' is given\n";
        }
    } else
        sema.error(to()) << "invocation not done on function type but instead type '" << to()->type() << "' is given\n";

    return sema.types.type_error();
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const {
    decl()->check(sema);

    if (const Expr* init_expr = init())
        init_expr->check(sema);
}

void ExprStmt::check(Sema& sema) const {
    expr()->check(sema);
}

static bool checkCond(Sema& sema, const Expr* cond) {
    if (cond->check(sema)->is_bool())
        return true;

    sema.error(cond) << "condition not a bool\n";
    return false;
}

void IfElseStmt::check(Sema& sema) const {
    checkCond(sema, cond());
    thenStmt()->check(sema);
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
    step()->check(sema);
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
    if (!fct()->continuation()) {
        const Pi* pi = fct()->pi();

        if (!pi->ret()->is_noret()) {
            if (pi->ret() == expr()->check(sema))
                return;
            else
                sema.error(expr()) << "expected return type '" << pi->ret() 
                    << "' but return expression is of type '" << expr()->type() << "'\n";
        } else
            sema.error(this) << "return statement not allowed for calling a continuation\n";
    } else
        sema.error(this) << "continuation is not allowed to use 'return'\n";
}

void ScopeStmt::check(Sema& sema) const {
    sema.pushScope();

    for_all (const &s, stmts())
        s->check(sema);

    sema.popScope();
}

} // namespace impala
