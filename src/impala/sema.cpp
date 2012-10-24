#include "impala/ast.h"

#include <stack>
#include <boost/unordered_map.hpp>

#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

#include "impala/dump.h"
#include "impala/type.h"

using anydsl2::Array;
using anydsl2::Location;
using anydsl2::Symbol;
using anydsl2::Type;
using anydsl2::Sigma;
using anydsl2::Pi;

namespace impala {

//------------------------------------------------------------------------------

class Sema {
public:

    Sema(World& world);
    ~Sema();

    /** 
     * @brief Looks up the current definition of \p sym.
     * 
     * @param sym The \p Symbol to look up.
     * 
     * @return Returns 0 on failure.
     */
    const Decl* lookup(Symbol symbol);

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
    const Decl* clash(Symbol symbol) const;

    /// Open a new scope.
    void pushScope() { ++depth_; }
    /// Discard current scope.
    void popScope();

    bool result() const { return result_; }

    std::ostream& error(const ASTNode* n) { result_ = false; return n->error(); }
    std::ostream& warning(const ASTNode* n) { return n->warning(); }

    World& world;

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
    typedef boost::unordered_map<Symbol, SlotStack*> Scope;

    bool result_;
    Scope scope_;
    int depth_;

#ifndef NDEBUG
    int refcounter_;
#endif
};

//------------------------------------------------------------------------------

Sema::Sema(World& world)
    : world(world)
    , result_(true)
    , depth_(0)
#ifndef NDEBUG
    , refcounter_(0)
#endif
{
    pushScope();
}

Sema::~Sema() {
    assert(depth_ == 1 && "root scope must be 1");
    popScope();

#ifndef NDEBUG
    assert(refcounter_ == 0 && "memory leak");
#endif
}

const Decl* Sema::lookup(Symbol sym) {
    Scope::iterator i = scope_.find(sym);
    if (i != scope_.end())
        return i->second->top().decl;
    else
        return 0;
}

void Sema::insert(const Decl* decl) {
    Symbol symbol = decl->symbol();

    assert(clash(symbol) == 0 && "must not be found");

    // create stack if necessary
    Scope::iterator i = scope_.find(symbol);
    if (i == scope_.end()) {
        // isn't C++ beautiful? We should switch to C++11 just because of auto
        std::pair<Scope::iterator, bool> p =
            scope_.insert(std::make_pair(symbol, new SlotStack()));
#ifndef NDEBUG
        ++refcounter_;
#endif
        i = p.first;
    }

    // get pointer to stack linked to symbol and push new Decl
    SlotStack* stack = i->second;
    stack->push(Slot(decl, depth_));
}

const Decl* Sema::clash(Symbol symbol) const {
    Scope::const_iterator i = scope_.find(symbol);
    if (i != scope_.end() && i->second->top().depth == depth_)
        return i->second->top().decl;
    else
        return 0;
}

void Sema::popScope() {
    assert(depth_ > 0 && "illegal depth value");

    Scope::iterator i = scope_.begin(); 
    while (i != scope_.end()) {
        SlotStack* stack = i->second;
        assert(!stack->empty() && "must have at least on element");
        
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

bool check(World& world, const Prg* prg) {
    Sema sema(world);
    prg->check(sema);
    return sema.result();
}

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) const {
    for_all (f, fcts())
        f->decl()->check(sema);

    for_all (f, fcts())
        f->check(sema);
}

void Lambda::check(Sema& sema) const {
    sema.pushScope();

    for_all (f, body()->fcts())
        f->decl()->check(sema);

    for_all (p, params())
        p->check(sema);

    for_all (s, body()->stmts())
        s->check(sema);

    sema.popScope();
}

void Fct::check(Sema& sema) const {
    lambda().check(sema);
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
    return sema.world.unit();
}

const Type* Literal::vcheck(Sema& sema) const {
    return sema.world.type(Token::literal2type((Token::Kind) kind()));
}

const Type* LambdaExpr::vcheck(Sema& sema) const {
    lambda().check(sema);
    return lambda().pi();
}

const Type* Tuple::vcheck(Sema& sema) const {
    Array<const Type*> elems(ops().size());

    size_t i = 0;
    for_all (op, ops())
        elems[i++] = op->check(sema);

    return sema.world.sigma(elems);
}

const Type* Id::vcheck(Sema& sema) const {
    if (const Decl* decl = sema.lookup(symbol())) {
        decl_ = decl;
        return decl->type();
    }

    sema.error(this) << "symbol '" << symbol() << "' not found in current scope\n";
    return sema.world.type_error();
}

const Type* PrefixExpr::vcheck(Sema& sema) const {
    if (!rhs()->lvalue())
        sema.error(rhs()) << "lvalue required as operand\n";

    return rhs()->check(sema);
}

const Type* InfixExpr::vcheck(Sema& sema) const {
    if (lhs()->check(sema) == rhs()->check(sema)) {
        if (Token::is_rel((TokenKind) kind()))
            return sema.world.type_u1();

        if (Token::is_asgn((TokenKind) kind())) {
            if (!lhs()->lvalue())
                sema.error(lhs()) << "no lvalue on left-hand side of assignment\n";
        }

        if (lhs()->type()->isa<TypeError>())
            return rhs()->type();
        else
            return lhs()->type();
    } else {
        sema.error(this) << "incompatible types in binary expression: '" 
            << lhs()->type() << "' and '" << rhs()->type() << "'\n";
    }

    return sema.world.type_error();
}

const Type* PostfixExpr::vcheck(Sema& sema) const {
    if (!lhs()->lvalue())
        sema.error(lhs()) << "lvalue required as operand\n";

    return lhs()->check(sema);
}

const Type* IndexExpr::vcheck(Sema& sema) const {
    if (const Sigma* sigma = lhs()->check(sema)->isa<Sigma>()) {
        if (is_int(index()->check(sema))) {
            if (const Literal* literal = index()->isa<Literal>()) {
                unsigned pos;

                switch (literal->kind()) {
#define IMPALA_LIT(itype, atype) \
                    case Literal::LIT_##itype: pos = literal->box().get_##atype(); break;
#include "impala/tokenlist.h"
                    default: ANYDSL2_UNREACHABLE;
                }

                if (pos < sigma->num_elems())
                    return sigma->elems()[pos];
                else
                    sema.error(index()) << "index (" << pos << ") out of bounds (" << sigma->num_elems() << ")\n";
            } else
                sema.error(index()) << "indexing expression must be a literal\n";
        } else
            sema.error(index()) << "indexing expression must be of integer type\n";
    } else
        sema.error(lhs()) << "left-hand side of index expression must be of sigma type\n";

    return sema.world.type_error();
}

const Type* Call::vcheck(Sema& sema) const { 
    if (const Pi* to_pi = to()->check(sema)->isa<Pi>()) {
        Array<const Type*> op_types(num_args() + 1); // reserve one more for return type

        for (size_t i = 0, e = num_args(); i != e; ++i)
            op_types[i] = arg(i)->check(sema);

        const Pi* pi;
        const Type* ret_type = to_pi->num_elems() == num_args() ? sema.world.noret() : return_type(to_pi);

        if (ret_type->isa<NoRet>())
            pi = sema.world.pi(op_types.slice_front(op_types.size()-1));
        else {
            op_types.back() = sema.world.pi1(ret_type);
            pi = sema.world.pi(op_types);
        }

        if (pi == to_pi)
            return ret_type;
        else {
            sema.error(to()) << "'" << to() << "' expects an invocation of type '" << to_pi 
                << "' but an invocation of type '" << pi << "' is given\n";
        }
    } else
        sema.error(to()) << "invocation not done on function type but instead type '" << to()->type() << "' is given\n";

    return sema.world.type_error();
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const {
    decl()->check(sema);

    if (const Expr* init_expr = init()) {
        if (decl()->type() != init_expr->check(sema))
            sema.error(this) << "initializing expression of type '" << init_expr->type() << "' but '" 
                << decl()->symbol() << "' declared of type '" << decl()->type() << '\n';
    }
}

void ExprStmt::check(Sema& sema) const {
    expr()->check(sema);
}

static bool checkCond(Sema& sema, const Expr* cond) {
    if (is_u1(cond->check(sema)))
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
    if (!lambda()->is_continuation()) {
        const Pi* pi = lambda()->pi();

        if (!return_type(pi)->isa<NoRet>()) {
            if (return_type(pi) == expr()->check(sema))
                return;
            else
                sema.error(expr()) << "expected return type '" << return_type(pi) 
                    << "' but return expression is of type '" << expr()->type() << "'\n";
        } else
            sema.error(this) << "return statement not allowed for calling a continuation\n";
    } else
        sema.error(this) << "continuation is not allowed to use 'return'\n";
}

void FctStmt::check(Sema& sema) const { fct()->check(sema); }

void ScopeStmt::check(Sema& sema) const {
    sema.pushScope();

    for_all (const &s, stmts())
        s->check(sema);

    sema.popScope();
}

} // namespace impala
