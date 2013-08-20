#include "impala/ast.h"

#include <vector>
#include <unordered_map>

#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/push.h"

#include "impala/dump.h"
#include "impala/type.h"

using namespace anydsl2;

namespace impala {

//------------------------------------------------------------------------------

class Sema {
public:
    Sema(TypeTable& typetable, bool nossa)
        : in_foreach_(false)
        , typetable_(typetable)
        , cur_fun_(nullptr)
        , result_(true)
        , nossa_(nossa)
    {}

    /** 
     * @brief Looks up the current definition of \p sym.
     * @return Returns 0 on failure.
     */
    const Decl* lookup(Symbol symbol) const;

    /** 
     * @brief Maps \p decl's symbol to \p decl.
     * 
     * If sym already has a definition in this scope an assertion is raised.
     * use \p clash in order to check this.
     */
    void insert(const Decl* decl);

    /** 
     * @brief Checks whether there already exists a \p Symbol \p symbol in the \em current scope.
     * @param symbol The \p Symbol to check.
     * @return The current mapping if the lookup succeeds, 0 otherwise.
     */
    const Decl* clash(Symbol symbol) const;

    void push_scope() { levels_.push_back(decl_stack_.size()); }
    void pop_scope();
    size_t depth() const { return levels_.size(); }
    bool result() const { return result_; }
    bool nossa() const { return nossa_; }
    std::ostream& error(const ASTNode* n) { result_ = false; return n->error(); }
    std::ostream& error(const Location& loc) { result_ = false; return loc.error(); }
    TypeTable& typetable() const { return typetable_; }
    void check(const Scope*) ;
    const Type* check(const Decl* decl) { decl->check(*this); return decl->refined_type_; }
    const Type* check(const Expr* expr) { assert(!expr->type_); return expr->type_ = expr->check(*this); }
    void check(const Stmt* stmt) { stmt->check(*this); }
    bool check_cond(const Expr* cond) {
        if (check(cond)->is_bool())
            return true;
        error(cond) << "condition not a bool\n";
        return false;
    }
    GenericMap copy_generic_map() { return cur_fun_ != nullptr ? cur_fun_->generic_map_ : GenericMap(); }
    GenericBuilder copy_generic_builder() { return cur_fun_ != nullptr ? cur_fun_->generic_builder_ : GenericBuilder(typetable_); }
    GenericMap* generic_map() const { return cur_fun_ != nullptr ? &cur_fun_->generic_map_ : nullptr; }
    GenericBuilder* generic_builder() const { return cur_fun_ != nullptr ? &cur_fun_->generic_builder_ : nullptr; }
    const Fun* cur_fun() const { return cur_fun_; }

    bool in_foreach_;

private:
    TypeTable& typetable_;
    const Fun* cur_fun_;
    bool result_;
    bool nossa_;

    typedef std::unordered_map<Symbol, const Decl*> Sym2Decl;
    Sym2Decl sym2decl_;
    std::vector<const Decl*> decl_stack_;
    std::vector<size_t> levels_;

    friend class Fun;
};

//------------------------------------------------------------------------------

const Decl* Sema::lookup(Symbol sym) const {
    auto i = sym2decl_.find(sym);
    return i != sym2decl_.end() ? i->second : 0;
}

void Sema::insert(const Decl* decl) {
    if (const Decl* other = clash(decl->symbol())) {
        error(decl) << "symbol '" << decl->symbol() << "' already defined\n";
        error(other) << "previous location here\n";
        return;
    } 

    Symbol symbol = decl->symbol();
    assert(clash(symbol) == 0 && "must not be found");

    auto i = sym2decl_.find(symbol);
    decl->shadows_ = i != sym2decl_.end() ? i->second : 0;
    decl->depth_ = depth();

    decl_stack_.push_back(decl);
    sym2decl_[symbol] = decl;
}

const Decl* Sema::clash(Symbol symbol) const {
    auto i = sym2decl_.find(symbol);
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

void Fun::refine(Sema& sema) const {
    generic_builder_ = sema.copy_generic_builder();
    generic_map_     = sema.copy_generic_map();
    sema.insert(this);

    sema.push_scope();
    ANYDSL2_PUSH(sema.cur_fun_, this);
    for (auto type_decl : generics()) {
        sema.check(type_decl);
    }

    refined_type_ = orig_type()->refine(sema);
    sema.pop_scope();
}

void Sema::check(const Scope* scope) {
    for (auto stmt : scope->stmts()) {
        if (auto fun_stmt = stmt->isa<FunStmt>())
            fun_stmt->fun()->refine(*this);
    }

    for (auto stmt : scope->stmts())
        check(stmt);
}

const Type* IdType::refine(const Sema& sema) const { 
    if (auto generic_builder = sema.generic_builder()) {
        if (auto decl = sema.lookup(Symbol(name.c_str()))) {
            if (auto type_decl = decl->isa<TypeDecl>()) {
                auto generic = generic_builder->use(type_decl->handle());
                if (type_decl->fun() == sema.cur_fun())
                    return generic;
                return typetable_.genericref(type_decl->fun(), generic);
            }
        }
    }
    return typetable_.type_error();
}

/*
 * Decl
 */

void VarDecl::check(Sema& sema) const {
    sema.insert(this);
    refined_type_ = orig_type_->refine(sema);
}

void Fun::check(Sema& sema) const {
    sema.push_scope();
    ANYDSL2_PUSH(sema.cur_fun_, this);

    if (refined_type_ == nullptr)
        refine(sema);

    for (auto type_decl : generics())
        sema.insert(type_decl);
    for (auto param : params()) 
        sema.check(param);
    sema.check(body());

    sema.pop_scope();
}

void TypeDecl::check(Sema& sema) const {
    sema.insert(this);
    handle_ = sema.cur_fun()->generic_builder_.new_def();
    fun_ = sema.cur_fun();
}

void Proto::check(Sema& sema) const {
    refined_type_ = orig_type_->refine(sema);
}

/*
 * Expr
 */

const Type* EmptyExpr::check(Sema& sema) const { return sema.typetable().type_void(); }
const Type* Literal::check(Sema& sema) const { return sema.typetable().primtype(literal2type()); }
const Type* FunExpr::check(Sema& sema) const { sema.check(fun()); return fun()->refined_fntype(); }

const Type* Tuple::check(Sema& sema) const {
    Array<const Type*> elems(ops().size());
    for (size_t i = 0, e = elems.size(); i != e; ++i)
        elems[i] = sema.check(op(i));

    return sema.typetable().tupletype(elems);
}

const Type* Id::check(Sema& sema) const {
    if (const Decl* decl = sema.lookup(symbol())) {
        decl_ = decl;

        if (sema.nossa() || sema.in_foreach_) {
            if (const VarDecl* vardecl = decl->isa<VarDecl>()) {
                if (!vardecl->refined_type()->isa<FnType>() && !vardecl->refined_type()->is_generic())
                    vardecl->is_address_taken_ = true;
            }
        }

        return decl->refined_type();
    }

    sema.error(this) << "symbol '" << symbol() << "' not found in current scope\n";
    return sema.typetable().type_error();
}

const Type* PrefixExpr::check(Sema& sema) const {
    switch (kind()) {
        case INC:
        case DEC:
            if (!rhs()->is_lvalue())
                sema.error(rhs()) << "lvalue required as operand\n";
            return sema.check(rhs());
        case L_N:
            if (!sema.check(rhs())->is_bool())
                sema.error(rhs()) << "logical not expects 'bool'\n";
            return sema.typetable().type_bool();
        default:
            return sema.check(rhs());
    }
}

const Type* InfixExpr::check(Sema& sema) const {
    if (Token::is_assign((TokenKind) kind())) {
        if (!lhs()->is_lvalue())
            sema.error(lhs()) << "no lvalue on left-hand side of assignment\n";
        else if (sema.check(lhs()) == sema.check(rhs()))
            return lhs()->type();
        else
            sema.error(this) << "incompatible types in assignment: '" 
                << lhs()->type() << "' and '" << rhs()->type() << "'\n";
    } else if (sema.check(lhs())->isa<PrimType>()) {
        if (sema.check(rhs())->isa<PrimType>()) {
            if (lhs()->type() == rhs()->type()) {
                if (Token::is_rel((TokenKind) kind()))
                    return sema.typetable().type_bool();

                if (kind() == L_A || kind() == L_O) {
                    if (!lhs()->type()->is_bool())
                        sema.error(this) << "logical binary expression expects 'bool'\n";
                    return sema.typetable().type_bool();
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

    return sema.typetable().type_error();
}

const Type* PostfixExpr::check(Sema& sema) const {
    if (!lhs()->is_lvalue())
        sema.error(lhs()) << "lvalue required as operand\n";

    return sema.check(lhs());
}

const Type* ConditionalExpr::check(Sema& sema) const {
    if (sema.check(cond())->is_bool()) {
        if (sema.check(t_expr()) == sema.check(f_expr()))
            return t_expr()->type();
        else
            sema.error(this) << "incompatible types in conditional expression\n";
    } else
        sema.error(cond()) << "condition not a bool\n";

    return t_expr()->type()->isa<TypeError>() ? f_expr()->type() : t_expr()->type();
}

const Type* IndexExpr::check(Sema& sema) const {
    if (auto tuple = sema.check(lhs())->isa<TupleType>()) {
        if (sema.check(index())->is_int()) {
            if (const Literal* literal = index()->isa<Literal>()) {
                unsigned pos;

                switch (literal->kind()) {
#define IMPALA_LIT(itype, atype) \
                    case Literal::LIT_##itype: pos = (unsigned) literal->box().get_##atype(); break;
#include "impala/tokenlist.h"
                    default: ANYDSL2_UNREACHABLE;
                }

                if (pos < tuple->size())
                    return tuple->elem(pos);
                else
                    sema.error(index()) << "index (" << pos << ") out of bounds (" << tuple->size() << ")\n";
            } else
                sema.error(index()) << "indexing expression must be a literal\n";
        } else
            sema.error(index()) << "indexing expression must be of integer type\n";
    } else
        sema.error(lhs()) << "left-hand side of index expression must be of tuple type\n";

    return sema.typetable().type_error();
}

const Type* Call::check(Sema& sema) const { 
    if (auto to_fn = sema.check(to())->isa<FnType>()) {
        Array<const Type*> op_types(num_args() + 1); // reserve one more for return type

        for (size_t i = 0, e = num_args(); i != e; ++i)
            op_types[i] = sema.check(arg(i));

        const FnType* call_fn;
        const Type* ret_type = to_fn->size() == num_args() ? sema.typetable().noret() : to_fn->return_type();

        if (ret_type->isa<NoRet>())
            call_fn = sema.typetable().fntype(op_types.slice_front(op_types.size()-1));
        else {
            op_types.back() = sema.typetable().fntype1(ret_type);
            call_fn = sema.typetable().fntype(op_types);
        }

        if (to_fn->check_with(call_fn)) {
            GenericMap map = sema.copy_generic_map();
            if (to_fn->infer_with(map, call_fn)) {
                auto result = ret_type->specialize(map);
                assert(result);
                return result;
            } else {
                sema.error(this->args_location()) << "cannot infer type '" << call_fn << "' induced by arguments\n";
                sema.error(to()) << "to invocation type '" << to_fn << "' with [" << map << "]\n";
            }
        } else {
            sema.error(to()) << "'" << to() << "' expects an invocation of type '" << to_fn 
                << "' but the invocation type '" << call_fn << "' is structural different\n";
        }
    } else
        sema.error(to()) << "invocation not done on function type but instead type '" << to()->type() << "' is given\n";

    return sema.typetable().type_error();
}

/*
 * Stmt
 */

void DeclStmt::check(Sema& sema) const { sema.check(decl()); }

void InitStmt::check(Sema& sema) const {
    if (sema.check(var_decl())->check_with(sema.check(init()))) {
        GenericMap map = sema.copy_generic_map();
        if (var_decl()->refined_type()->infer_with(map, init()->type()))
            return;
        else {
            sema.error(init()) << "cannot infer initializing type '" << init()->type() << "'\n";
            sema.error(var_decl()) << "to declared type '" << var_decl()->refined_type() << "' with '" << map << "'\n";
        }
    } else {
        sema.error(this) << "initializing expression of type '" << init()->type() << "' but '" 
            << var_decl()->symbol() << "' declared of type '" << var_decl()->refined_type() << '\n';
    }
}

void ExprStmt::check(Sema& sema) const {
    sema.check(expr());
}

void IfElseStmt::check(Sema& sema) const {
    sema.check_cond(cond());
    sema.check(then_scope());
    sema.check(else_scope());
}

void DoWhileStmt::check(Sema& sema) const {
    sema.check(body());
    sema.check_cond(cond());
}

void ForStmt::check(Sema& sema) const {
    sema.push_scope();
    sema.check(init());
    sema.check_cond(cond());
    sema.check(step());
    sema.check(body());
    sema.pop_scope();
}

void ForeachStmt::check(Sema& sema) const {
#if 0
    sema.push_scope();

    const Type* left_type_;
    if (init_decl()) {
        sema.insert(init_decl());
        left_type_ = init_decl()->type();
    } else {
        left_type_ = sema.check(init_expr());
    }

    // generator call
    if (auto to_fn = sema.check(call()->to())->isa<FnType>()) {
        // reserve one for the body type and one for the next continuation type
        Array<const Type*> op_types(call()->num_args() + 1 + 1);

        for (size_t i = 0, e = call()->num_args(); i != e; ++i)
            op_types[i] = sema.check(call()->arg(i));
        
        const Type* elems[2] = { left_type_, sema.typetable().fntype0() };
        op_types[call()->num_args()] = fntype_ = sema.typetable().fntype(elems);
        op_types.back() = sema.typetable().fntype0();    
        const FnType* call_fn = sema.typetable().fntype(op_types);

        if (to_fn->check_with(call_fn)) {
            GenericMap map = sema.copy_generic_map();
            if (!to_fn->infer_with(map, call_fn)) {
                sema.error(call()->args_location()) << "cannot infer type '" << call_fn << "' induced by arguments\n";
                sema.error(call()->to()) << "to invocation type '" << to_fn << "' with '" << map << "'\n";
            }
        } else {
            sema.error(call()->to()) << "'" << call()->to() << "' expects an invocation of type '" << to_fn 
                << "' but the invocation type '" << call_fn << "' is structural different\n";
        }
    } else
        sema.error(call()->to()) << "invocation not done on function type but instead type '" 
            << call()->to()->type() << "' is given\n";

    ANYDSL2_PUSH(sema.in_foreach_, true);

    if (const ScopeStmt* scope = body()->isa<ScopeStmt>())
        sema.check_stmts(scope);
    else
        sema.check(body());

    if (init_decl())
        init_decl()->is_address_taken_ = false;

    sema.pop_scope();
#endif
}

void BreakStmt::check(Sema& sema) const {
    if (!loop() && !sema.in_foreach_)
        sema.error(this) << "break statement not within a loop\n";
}

void ContinueStmt::check(Sema& sema) const {
    if (!loop() && !sema.in_foreach_)
        sema.error(this) << "continue statement not within a loop\n";
}

void ReturnStmt::check(Sema& sema) const {
    if (!fun()->is_continuation()) {
        const Type* ret_type = fun()->refined_fntype()->return_type();

        if (ret_type->isa<Void>()) {
            if (!expr())
                return;
            else
                sema.error(expr()) << "return expression in a function returning 'void'\n";
        } else if (!sema.check(expr())->isa<NoRet>()) {
            if (expr()->type()->isa<TypeError>())
                return;
            if (ret_type->check_with(expr()->type())) {
                GenericMap map = sema.copy_generic_map();
                if (ret_type->infer_with(map, expr()->type()))
                    return;
                else
                    sema.error(expr()) << "cannot infer type '" << expr()->type() 
                        << "' of return expression to return type '" << ret_type << "' with [" << map << "]\n";
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
    sema.check(scope());
    sema.pop_scope();
}

void FunStmt::check(Sema& sema) const {
    sema.check(fun());
}

//------------------------------------------------------------------------------

bool check(TypeTable& typetable, const Scope* prg, bool nossa) { Sema sema(typetable, nossa); sema.check(prg); return sema.result(); }

//------------------------------------------------------------------------------

} // namespace impala
