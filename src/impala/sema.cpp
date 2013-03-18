#include "impala/ast.h"

#include <vector>
#include <boost/unordered_map.hpp>

#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

#include "impala/dump.h"
#include "impala/type.h"

using namespace anydsl2;

namespace impala {

//------------------------------------------------------------------------------

class Sema {
public:

    Sema(World& world)
        : world_(world)
        , result_(true)
    {}

    /** 
     * @brief Looks up the current definition of \p sym.
     * @return Returns 0 on failure.
     */
    const Decl* lookup(Symbol symbol);

    /** 
     * @brief Maps \p decl's symbol to \p decl.
     * 
     * If sym already has a definition in this scope an assertion is raised.
     * use \p clash in order to check this.
     */
    void insert(const Decl* decl);

    /** 
     * @brief Checks whether there already exists a \p Symbol \p sym in the \em current scope.
     * @param sym The \p Symbol to check.
     * @return The current mapping if the lookup succeeds, 0 otherwise.
     */
    const Decl* clash(Symbol symbol) const;

    void push_scope() { levels_.push_back(decl_stack_.size()); }
    void pop_scope();
    size_t depth() const { return levels_.size(); }
    bool result() const { return result_; }
    std::ostream& error(const ASTNode* n) { result_ = false; return n->error(); }
    std::ostream& error(const Location& loc) { result_ = false; return loc.error(); }
    GenericMap fill_map();
    World& world() { return world_; }

    std::vector< boost::unordered_set<const Generic*> > bound_generics_;

private:

    World& world_;
    bool result_;

    typedef boost::unordered_map<Symbol, const Decl*> Sym2Decl;
    Sym2Decl sym2decl_;
    std::vector<const Decl*> decl_stack_;
    std::vector<size_t> levels_;
};

//------------------------------------------------------------------------------

const Decl* Sema::lookup(Symbol sym) {
    Sym2Decl::iterator i = sym2decl_.find(sym);
    return i != sym2decl_.end() ? i->second : 0;
}

void Sema::insert(const Decl* decl) {
    Symbol symbol = decl->symbol();
    assert(clash(symbol) == 0 && "must not be found");

    Sym2Decl::iterator i = sym2decl_.find(symbol);
    decl->shadows_ = i != sym2decl_.end() ? i->second : 0;
    decl->depth_ = depth();

    decl_stack_.push_back(decl);
    sym2decl_[symbol] = decl;
}

const Decl* Sema::clash(Symbol symbol) const {
    Sym2Decl::const_iterator i = sym2decl_.find(symbol);
    if (i == sym2decl_.end())
        return 0;

    const Decl* decl = i->second;
    return (decl && decl->depth() == depth()) ? decl : 0;
}

void Sema::pop_scope() {
    size_t level = levels_.back();
    for (size_t i = level, e = decl_stack_.size(); i != e; ++i) {
        const Decl* decl = decl_stack_[i];
        sym2decl_[decl->symbol()] = decl->shadows();
    }

    decl_stack_.resize(level);
    levels_.pop_back();
}

//------------------------------------------------------------------------------

bool check(World& world, const Prg* prg) {
    Sema sema(world);
    prg->check(sema);
    return sema.result();
}

//------------------------------------------------------------------------------

void Prg::check(Sema& sema) const {
    for_all (f, named_funs())
        f->insert(sema);

    for_all (f, named_funs())
        f->check(sema);
}

static void propagate_set(const Type* type, boost::unordered_set<const Generic*>& bound) {
    for_all (elem, type->elems())
        if (const Generic* generic = elem->isa<Generic>())
            bound.insert(generic);
        else 
            propagate_set(elem, bound);
}

GenericMap Sema::fill_map() {
    GenericMap map;
    for_all (set, bound_generics_)
        for_all (generic, set)
            map[generic] = generic;
    return map;
}

void Fun::fun_check(Sema& sema) const {
    sema.push_scope();
    boost::unordered_set<const Generic*> bound;
    propagate_set(pi(), bound);
    sema.bound_generics_.push_back(bound);

    for_all (f, body()->named_funs())
        f->insert(sema);

    for_all (p, params())
        p->insert(sema);

    for_all (s, body()->stmts())
        s->check(sema);

    sema.bound_generics_.pop_back();
    sema.pop_scope();
}

void Decl::insert(Sema& sema) const {
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
    return sema.world().unit();
}

const Type* Literal::vcheck(Sema& sema) const {
    return sema.world().type(literal2type());
}

const Type* FunExpr::vcheck(Sema& sema) const {
    fun_check(sema);
    return pi();
}

const Type* Tuple::vcheck(Sema& sema) const {
    Array<const Type*> elems(ops().size());
    for_all2 (&elem, elems, op, ops())
        elem = op->check(sema);

    return sema.world().sigma(elems);
}

const Type* Id::vcheck(Sema& sema) const {
    if (const Decl* decl = sema.lookup(symbol())) {
        decl_ = decl;
        return decl->type();
    }

    sema.error(this) << "symbol '" << symbol() << "' not found in current scope\n";
    return sema.world().type_error();
}

const Type* PrefixExpr::vcheck(Sema& sema) const {
    switch (kind()) {
        case INC:
        case DEC:
            if (!rhs()->is_lvalue())
                sema.error(rhs()) << "lvalue required as operand\n";
            return rhs()->check(sema);
        case L_N:
            if (!rhs()->check(sema)->is_u1())
                sema.error(rhs()) << "logical not expects 'bool'\n";
            return sema.world().type_u1();
        default:
            return rhs()->check(sema);
    }
}

const Type* InfixExpr::vcheck(Sema& sema) const {
    if (Token::is_assign((TokenKind) kind())) {
        if (!lhs()->is_lvalue())
            sema.error(lhs()) << "no lvalue on left-hand side of assignment\n";
        else if (lhs()->check(sema) == rhs()->check(sema))
            return lhs()->type();
        else
            sema.error(this) << "incompatible types in assignment: '" 
                << lhs()->type() << "' and '" << rhs()->type() << "'\n";
    } else if (lhs()->check(sema)->is_primtype()) {
        if (rhs()->check(sema)->is_primtype()) {
            if (lhs()->type() == rhs()->type()) {
                if (Token::is_rel((TokenKind) kind()))
                    return sema.world().type_u1();

                if (kind() == L_A || kind() == L_O) {
                    if (!lhs()->type()->is_u1())
                        sema.error(this) << "logical binary expression expects 'bool'\n";
                    return sema.world().type_u1();
                }

                if (lhs()->type()->isa<TypeError>())
                    return rhs()->type();
                else
                    return lhs()->type();
            } else {
                sema.error(this) << "incompatible types in binary expression: '" 
                    << lhs()->type() << "' and '" << rhs()->type() << "'\n";
            }
        } else
            sema.error(lhs()) << "primitive type expected on right-hand side of binary expressions\n";
    } else
        sema.error(lhs()) << "primitive type expected on left-hand side of binary expressions\n";

    return sema.world().type_error();
}

const Type* PostfixExpr::vcheck(Sema& sema) const {
    if (!lhs()->is_lvalue())
        sema.error(lhs()) << "lvalue required as operand\n";

    return lhs()->check(sema);
}

const Type* ConditionalExpr::vcheck(Sema& sema) const {
    if (cond()->check(sema)->is_u1()) {
        if (t_expr()->check(sema) == f_expr()->check(sema))
            return t_expr()->type();
        else
            sema.error(this) << "incompatible types in conditional expression\n";
    } else
        sema.error(cond()) << "condition not a bool\n";

    return t_expr()->type()->isa<TypeError>() ? f_expr()->type() : t_expr()->type();
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
                    default: ANYDSL2_UNREACHABLE;
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

    return sema.world().type_error();
}

const Type* Call::vcheck(Sema& sema) const { 
    if (const Pi* to_pi = to()->check(sema)->isa<Pi>()) {
        Array<const Type*> op_types(num_args() + 1); // reserve one more for return type

        for (size_t i = 0, e = num_args(); i != e; ++i)
            op_types[i] = arg(i)->check(sema);

        const Pi* call_pi;
        const Type* ret_type = to_pi->size() == num_args() ? sema.world().noret() : return_type(to_pi);

        if (ret_type->isa<NoRet>())
            call_pi = sema.world().pi(op_types.slice_front(op_types.size()-1));
        else {
            op_types.back() = sema.world().pi1(ret_type);
            call_pi = sema.world().pi(op_types);
        }

        if (to_pi->check_with(call_pi)) {
            GenericMap map = sema.fill_map();
            if (to_pi->infer_with(map, call_pi)) {
                if (const Generic* generic = ret_type->isa<Generic>())
                    return map[generic];
                else
                    return ret_type;
            } else {
                sema.error(this->args_location()) << "cannot infer type '" << call_pi << "' induced by arguments\n";
                sema.error(to()) << "to invocation type '" << to_pi << "' with '" << map << "'\n";
            }
        } else {
            sema.error(to()) << "'" << to() << "' expects an invocation of type '" << to_pi 
                << "' but the invocation type '" << call_pi << "' is structural different\n";
        }
    } else
        sema.error(to()) << "invocation not done on function type but instead type '" << to()->type() << "' is given\n";

    return sema.world().type_error();
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const {
    var_decl()->insert(sema);

    if (const Expr* init_expr = init()) {
        if (var_decl()->type()->check_with(init_expr->check(sema))) {
            GenericMap map = sema.fill_map();
            if (var_decl()->type()->infer_with(map, init_expr->type()))
                return;
            else {
                sema.error(init_expr) << "cannot infer initializing type '" << init_expr->type() << "'\n";
                sema.error(var_decl()) << "to declared type '" << var_decl()->type() << "' with '" << map << "'\n";
            }
        } else {
            sema.error(this) << "initializing expression of type '" << init_expr->type() << "' but '" 
                << var_decl()->symbol() << "' declared of type '" << var_decl()->type() << '\n';
        }
    }
}

void ExprStmt::check(Sema& sema) const {
    expr()->check(sema);
}

static bool check_cond(Sema& sema, const Expr* cond) {
    if (cond->check(sema)->is_u1())
        return true;

    sema.error(cond) << "condition not a bool\n";
    return false;
}

void IfElseStmt::check(Sema& sema) const {
    check_cond(sema, cond());
    then_stmt()->check(sema);
    else_stmt()->check(sema);
}

void DoWhileStmt::check(Sema& sema) const {
    body()->check(sema);
    check_cond(sema, cond());
}

void ForStmt::check(Sema& sema) const {
    sema.push_scope();
    init()->check(sema);
    check_cond(sema, cond());
    step()->check(sema);

    if (const ScopeStmt* scope = body()->isa<ScopeStmt>())
        scope->check_stmts(sema);
    else
        body()->check(sema);

    sema.pop_scope();
}

void ForeachStmt::check(Sema& sema) const {
    // TODO
    sema.push_scope();
    init()->check(sema);
    
    // generator call
    if (const Pi* to_pi = to()->check(sema)->isa<Pi>()) {
        Array<const Type*> op_types(num_args() + 1 + 2); // reserve one more for return type

        for (size_t i = 0, e = num_args(); i != e; ++i)
            op_types[i] = arg(i)->check(sema);
            
        // add stuff
        op_types[num_args()] = sema.world().generic(0);
        
        std::vector<const Type*> elems;
        elems.push_back(sema.world().type_u32());
        elems.push_back(sema.world().generic(0));
        std::vector<const Type*> inner_elems;
        inner_elems.push_back(sema.world().generic(0));
        elems.push_back(sema.world().pi(inner_elems));
        op_types[num_args() + 1] = sema.world().pi(elems);
        
        //op_types[num_args() + 1] = to_pi->elem(num_args() + 1);
        
        /*std::vector<const Type*> elems;
        elems.push_back(cg.world().type_u32());
        op_types[num_args() + 1] = cg.world().pi(elems);*/
        //

        const Pi* call_pi;
        const Type* ret_type = to_pi->size() == num_args() ? sema.world().noret() : return_type(to_pi);

        if (ret_type->isa<NoRet>())
            call_pi = sema.world().pi(op_types.slice_front(op_types.size()-1));
        else {
            op_types.back() = sema.world().pi1(ret_type);
            call_pi = sema.world().pi(op_types);
        }

        if (to_pi->check_with(call_pi)) {
            GenericMap map = sema.fill_map();
            if (to_pi->infer_with(map, call_pi)) {
                /*if (const Generic* generic = ret_type->isa<Generic>())
                    return map[generic];
                else
                    return ret_type;*/
            } else {
                sema.error(this->args_location()) << "cannot infer type '" << call_pi << "' induced by arguments\n";
                sema.error(to()) << "to invocation type '" << to_pi << "' with '" << map << "'\n";
            }
        } else {
            sema.error(to()) << "'" << to() << "' expects an invocation of type '" << to_pi 
                << "' but the invocation type '" << call_pi << "' is structural different\n";
        }
    } else
        sema.error(to()) << "invocation not done on function type but instead type '" << to()->type() << "' is given\n";
    //
    

    if (const ScopeStmt* scope = body()->isa<ScopeStmt>())
        scope->check_stmts(sema);
    else
        body()->check(sema);

    sema.pop_scope();
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
    if (!fun()->is_continuation()) {
        const Pi* pi = fun()->pi();
        const Type* ret_type = return_type(pi);

        if (ret_type->isa<Void>()) {
            if (!expr())
                return;
            else
                sema.error(expr()) << "return expression in a function returning 'void'\n";
        } else if (!ret_type->isa<NoRet>()) {
            if (expr()->check(sema)->isa<TypeError>())
                return;
            if (ret_type->check_with(expr()->type())) {
                GenericMap map = sema.fill_map();
                if (ret_type->infer_with(map, expr()->type()))
                    return;
                else
                    sema.error(expr()) << "cannot infer type '" << expr()->type() 
                        << "' of return expression to return type '" << ret_type << "' with '" << map << "'\n";
            } else 
                sema.error(expr()) << "expected return type '" << ret_type 
                    << "' but return expression is of type '" << expr()->type() << "'\n";
        } else
            sema.error(this) << "return statement not allowed for calling a continuation\n";
    } else
        sema.error(this) << "continuation is not allowed to use 'return'\n";
}

void ScopeStmt::check(Sema& sema) const {
    sema.push_scope();
    check_stmts(sema);
    sema.pop_scope();
}

void ScopeStmt::check_stmts(Sema& sema) const {
    for_all (s, stmts())
        s->check(sema);
}

} // namespace impala
