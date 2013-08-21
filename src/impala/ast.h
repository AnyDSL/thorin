#ifndef IMPALA_AST_H
#define IMPALA_AST_H

#include <vector>

#include "anydsl2/util/array.h"
#include "anydsl2/util/assert.h"
#include "anydsl2/util/autoptr.h"
#include "anydsl2/util/box.h"
#include "anydsl2/util/cast.h"
#include "anydsl2/util/location.h"
#include "anydsl2/util/types.h"

#include "impala/token.h"
#include "impala/type.h"

namespace anydsl2 {
    class Def;
    class JumpTarget;
    class Lambda;
    class Param;
    class Ref;
}

namespace impala {

class CodeGen;
class Expr;
class Global;
class Printer;
class ScopeStmt;
class Sema;
class Stmt;
class VarDecl;
class TypeDecl;

typedef anydsl2::AutoVector<const VarDecl*> VarDecls;
typedef anydsl2::AutoVector<const Expr*> Exprs;
typedef anydsl2::AutoVector<const Stmt*> Stmts;
typedef anydsl2::AutoPtr<const anydsl2::Ref> RefPtr;
typedef anydsl2::AutoVector<const TypeDecl*> TypeDecls;

//------------------------------------------------------------------------------

class ASTNode : public anydsl2::HasLocation, public anydsl2::MagicCast {
public:
#ifndef NDEBUG
    virtual ~ASTNode() { assert(loc_.is_set()); }
#endif
    virtual std::ostream& print(Printer& p) const = 0;
    void dump() const;
};

class Scope : public ASTNode {
public:
    Scope() {}
    Scope(const anydsl2::Location& loc) {
        loc_ = loc;
    }

    const Stmts& stmts() const { return stmts_; }
    const Stmt* stmt(size_t i) const { return stmts_[i]; }
    bool empty() const { return stmts_.empty(); }
    virtual std::ostream& print(Printer& p) const;

private:
    mutable Stmts stmts_;

    friend class Parser;
};

//------------------------------------------------------------------------------

class Decl : public ASTNode {
public:
    Decl()
        : orig_type_(nullptr)
        , refined_type_(nullptr)
    {}

    anydsl2::Symbol symbol() const { return symbol_; }
    const Type* orig_type() const { return orig_type_; }
    const Type* refined_type() const { return refined_type_; }
    size_t depth() const { return depth_; }
    const Decl* shadows() const { return shadows_; }
    virtual void check(Sema& sema) const = 0;

protected:
    anydsl2::Symbol symbol_;
    const Type* orig_type_;
    mutable const Type* refined_type_;

private:
    mutable const Decl* shadows_;
    mutable size_t depth_;

    friend class Sema;
};

class Fun : public Decl {
public:
    Fun(TypeTable& typetable)
        : generic_builder_(typetable)
    {}

    const Scope* body() const { return body_; }
    const VarDecl* param(size_t i) const { return params_[i]; }
    const VarDecls& params() const { return params_; }
    const FnType* orig_fntype() const { return orig_type_->as<FnType>(); }
    const FnType* refined_fntype() const { return refined_type_->as<FnType>(); }
    const TypeDecls& generics() const { return generics_; }
    bool is_extern() const { return extern_; }
    bool is_continuation() const { return orig_fntype()->return_type()->isa<NoRet>() != nullptr; }
    anydsl2::Lambda* lambda() const { return lambda_; }
    const anydsl2::Param* ret_param() const { return ret_param_; }
    void check_head(Sema&) const;
    virtual void check(Sema& sema) const;
    virtual std::ostream& print(Printer& p) const;

private:
    VarDecls params_;
    anydsl2::AutoPtr<const Scope> body_;
    bool extern_;
    TypeDecls generics_;
    mutable anydsl2::Lambda* lambda_;
    mutable const anydsl2::Param* ret_param_;
    mutable GenericBuilder generic_builder_;
    mutable GenericMap generic_map_;

    friend class Parser;
    friend class Sema;
    friend class CodeGen;
    friend class TypeDecl;
    friend class FunExpr;
};

class TypeDecl : public Decl {
public:
    TypeDecl(const Token& tok)
        : handle_(-1)
    {
        symbol_ = tok.symbol();
        set_loc(tok.loc());
    }

    size_t handle() const { return handle_; }
    const Fun* fun() const { return fun_; }
    virtual void check(Sema& sema) const;
    virtual std::ostream& print(Printer& p) const;

private:

    mutable size_t handle_;
    mutable const Fun* fun_;

    friend class Fun;
    friend class Sema;
};

class VarDecl : public Decl {
public:
    VarDecl(size_t handle, const Token& tok, const Type* orig_type, const anydsl2::Position& pos2)
        : handle_(handle)
        , is_address_taken_(false)
    {
        symbol_ = tok.symbol();
        orig_type_ = orig_type;
        set_loc(tok.pos1(), pos2);
    }

    size_t handle() const { return handle_; }
    bool is_address_taken() const { return is_address_taken_; }
    const Fun* fun() const { return fun_; }
    virtual void check(Sema& sema) const;
    virtual std::ostream& print(Printer& p) const;

private:
    size_t handle_;
    mutable bool is_address_taken_;
    mutable const Fun* fun_;

    friend class Id;
    friend class ForeachStmt;
};

class Proto : public Decl {
public:
    Proto(anydsl2::Symbol symbol) {
        symbol_ = symbol;
    }

    const FnType* orig_fntype() const { return orig_type_->as<FnType>(); }
    const FnType* refined_fntype() const { return refined_type_->as<FnType>(); }
    virtual void check(Sema& sema) const;
    virtual std::ostream& print(Printer& p) const;

    friend class Parser;
};

//------------------------------------------------------------------------------

class Expr : public ASTNode {
public:
    Expr() 
        : type_(nullptr) 
    {}

    const Exprs& ops() const { return ops_; }
    const Expr* op(size_t i) const { return ops_[i]; }
    size_t size() const { return ops_.size(); }
    bool empty() const { return size() == 0; }
    const Type* type() const { return type_; }
    virtual bool is_lvalue() const = 0;

private:
    virtual RefPtr emit(CodeGen& cg) const = 0;
    virtual const Type* check(Sema& sema) const = 0;

protected:
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;

    Exprs ops_;
    mutable const Type* type_;

    friend class CodeGen;
    friend class Sema;
};

class EmptyExpr : public Expr {
public:
    EmptyExpr(const anydsl2::Location& loc) { loc_ = loc; }

    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
};

class Literal : public Expr {
public:
    enum Kind {
#define IMPALA_LIT(itype, atype) LIT_##itype = Token::LIT_##itype,
#include "impala/tokenlist.h"
        LIT_bool
    };

    Literal(const anydsl2::Location& loc, Kind kind, anydsl2::Box box)
        : kind_(kind)
        , box_(box)
    {
        loc_= loc;
    }

    Kind kind() const { return kind_; }
    anydsl2::Box box() const { return box_; }
    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;
    TokenKind literal2type() const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;

    Kind kind_;
    anydsl2::Box box_;
};

class FunExpr : public Expr {
public:
    FunExpr(TypeTable& typetable)
        : fun_(new Fun(typetable))
    {}

    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;
    const Fun* fun() const { return fun_; }

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;

    anydsl2::AutoPtr<Fun> fun_;

    friend class Parser;
};

class Tuple : public Expr {
public:
    Tuple(const anydsl2::Position& pos1) { loc_.set_pos1(pos1); }

    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;

    friend class Parser;
};

class Id : public Expr {
public:
    Id(const Token& tok)
        : symbol_(tok.symbol())
    {
        loc_ = tok.loc();
    }

    anydsl2::Symbol symbol() const { return symbol_; }
    const Decl* decl() const { return decl_; }

    virtual bool is_lvalue() const { return true; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;

    anydsl2::Symbol symbol_;
    mutable const Decl* decl_; ///< Declaration of the variable in use.
};

class PrefixExpr : public Expr {
public:
    enum Kind {
#define IMPALA_PREFIX(tok, str, prec) tok = Token:: tok,
#include "impala/tokenlist.h"
    };

    PrefixExpr(const anydsl2::Position& pos1, Kind kind, const Expr* rhs)
        : kind_(kind)
    {
        ops_.push_back(rhs);
        set_loc(pos1, rhs->pos2());
    }

    const Expr* rhs() const { return ops_[0]; }
    Kind kind() const { return kind_; }
    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;

    Kind kind_;
};

class InfixExpr : public Expr {
public:
    enum Kind {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) tok = Token:: tok,
#define IMPALA_INFIX(     tok, str, lprec, rprec) tok = Token:: tok,
#include "impala/tokenlist.h"
    };

    InfixExpr(const Expr* lhs, Kind kind, const Expr* rhs)
        : kind_(kind)
    {
        ops_.push_back(lhs);
        ops_.push_back(rhs);
        set_loc(lhs->pos1(), rhs->pos2());
    }

    const Expr* lhs() const { return ops_[0]; }
    const Expr* rhs() const { return ops_[1]; }
    Kind kind() const { return kind_; }
    virtual bool is_lvalue() const { return Token::is_assign((TokenKind) kind()); }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;

    Kind kind_;
};

/**
 * Just for expr++ and expr--.
 * For indexing and function calls use \p IndexExpr or \p Call, respectively.
 */
class PostfixExpr : public Expr {
public:
    enum Kind {
        INC = Token::INC,
        DEC = Token::DEC
    };

    PostfixExpr(const Expr* lhs, Kind kind, const anydsl2::Position& pos2)
        : kind_(kind)
    {
        ops_.push_back(lhs);
        set_loc(lhs->pos1(), pos2);
    }

    const Expr* lhs() const { return ops_[0]; }
    Kind kind() const { return kind_; }
    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;

    Kind kind_;
};

class ConditionalExpr : public Expr {
public:
    ConditionalExpr(const Expr* cond, const Expr* t_expr, const Expr* f_expr) {
        ops_.push_back(cond);
        ops_.push_back(t_expr);
        ops_.push_back(f_expr);
        set_loc(cond->pos1(), f_expr->pos2());
    }

    const Expr* cond()   const { return ops_[0]; }
    const Expr* t_expr() const { return ops_[1]; }
    const Expr* f_expr() const { return ops_[2]; }
    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
};

class IndexExpr : public Expr {
public:
    IndexExpr(const anydsl2::Position& pos1, const Expr* lhs, const Expr* index, const anydsl2::Position& pos2) {
        ops_.push_back(lhs);
        ops_.push_back(index);
        set_loc(pos1, pos2);
    }

    const Expr* lhs() const { return ops_[0]; }
    const Expr* index() const { return ops_[1]; }
    virtual bool is_lvalue() const { return true; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
};

class Call : public Expr {
public:
    Call(const Expr* fun) { ops_.push_back(fun); }

    void append_arg(const Expr* expr) { ops_.push_back(expr); }
    void set_pos2(const anydsl2::Position& pos2) {
        assert(!ops_.empty());
        HasLocation::set_loc(ops_.front()->pos1(), pos2);
    }
    const Expr* to() const { return ops_.front(); }
    size_t num_args() const { return size() - 1; }
    anydsl2::ArrayRef<const Expr*> args() const { return anydsl2::ArrayRef<const Expr*>(&*ops_.begin() + 1, num_args()); }
    const Expr* arg(size_t i) const { return op(i+1); }
    anydsl2::Location args_location() const;
    bool is_continuation_call() const;
    virtual bool is_lvalue() const { return false; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual const Type* check(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
};

//------------------------------------------------------------------------------

class Stmt : public ASTNode {
private:
    virtual void check(Sema& sema) const = 0;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const = 0;

    friend class Sema;
    friend class CodeGen;
};

class ExprStmt : public Stmt {
public:
    ExprStmt(const Expr* expr, const anydsl2::Position& pos2)
        : expr_(expr)
    {
        set_loc(expr->pos1(), pos2);
    }

    const Expr* expr() const { return expr_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Expr> expr_;
};

class DeclStmt : public Stmt {
public:
    DeclStmt(const Decl* decl)
        : decl_(decl)
    {}

    const Decl* decl() const { return decl_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Decl> decl_;
};

class InitStmt : public Stmt {
public:
    InitStmt(const VarDecl* var_decl, const Expr* init)
        : var_decl_(var_decl)
        , init_(init)
    {
        set_loc(var_decl->pos1(), init->pos2());
    }

    const VarDecl* var_decl() const { return var_decl_; }
    const Expr* init() const { return init_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const VarDecl> var_decl_;
    anydsl2::AutoPtr<const Expr> init_;
};

class IfElseStmt: public Stmt {
public:
    const Expr* cond() const { return cond_; }
    const Scope* then_scope() const { return then_scope_; }
    const Scope* else_scope() const { return else_scope_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Expr> cond_;
    anydsl2::AutoPtr<const Scope> then_scope_;
    anydsl2::AutoPtr<const Scope> else_scope_;

    friend class Parser;
};

class Loop : public Stmt {
public:
    Loop() {}
    const Expr* cond() const { return cond_; }
    const Scope* body() const { return body_; }

private:
    anydsl2::AutoPtr<const Expr> cond_;
    anydsl2::AutoPtr<const Scope> body_;

    friend class Parser;
};

class DoWhileStmt : public Loop {
public:
    DoWhileStmt() {}

    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;
};

class ForStmt : public Loop {
public:
    ForStmt() {}

    const Stmt* init() const { return init_; }
    const Expr* step() const { return step_; }
    bool is_while() const;
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Stmt> init_;
    anydsl2::AutoPtr<const Expr> step_;

    friend class Parser;
};


class ForeachStmt : public Stmt {
public:
    ForeachStmt() {}

    const Stmt* body() const { return body_; }
    const VarDecl* init_decl() const { return init_decl_; }
    const Expr* init_expr() const { return init_expr_; }
    const Call* call() const { return call_; }
    const ASTNode* init() const { return (const ASTNode*) ((uintptr_t) init_decl_.get() | (uintptr_t) init_expr_.get()); }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Stmt> body_;
    anydsl2::AutoPtr<const VarDecl> init_decl_;
    anydsl2::AutoPtr<const Expr> init_expr_;
    anydsl2::AutoPtr<const Call> call_;
    mutable const FnType* fntype_;

    friend class Parser;
};

class BreakStmt : public Stmt {
public:
    BreakStmt(const anydsl2::Position& pos1, const anydsl2::Position& pos2, const Loop* loop)
        : loop_(loop)
    {
        set_loc(pos1, pos2);
    }

    const Loop* loop() const { return loop_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    const Loop* loop_;
};

class ContinueStmt : public Stmt {
public:
    ContinueStmt(const anydsl2::Position& pos1, const anydsl2::Position& pos2, const Loop* loop)
        : loop_(loop)
    {
        set_loc(pos1, pos2);
    }

    const Loop* loop() const { return loop_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    const Loop* loop_;
};

class ReturnStmt : public Stmt {
public:
    ReturnStmt(const anydsl2::Position& pos1, const Expr* expr, const Fun* fun, const anydsl2::Position& pos2)
        : expr_(expr)
        , fun_(fun)
    {
        set_loc(pos1, pos2);
    }

    const Expr* expr() const { return expr_; }
    const Fun* fun() const { return fun_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Expr> expr_;
    const Fun* fun_;
};

class FunStmt : public Stmt {
public:
    FunStmt(TypeTable& typetable)
        : fun_(new Fun(typetable))
    {}

    const Fun* fun() const { return fun_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<Fun> fun_;

    friend class Parser;
};

class ScopeStmt : public Stmt {
public:
    ScopeStmt() {}
    ScopeStmt(const anydsl2::Location& loc) {
        loc_ = loc;
    }

    const Scope* scope() const { return scope_; }
    virtual std::ostream& print(Printer& p) const;

private:
    virtual void check(Sema& sema) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

    anydsl2::AutoPtr<const Scope> scope_;

    friend class Parser;
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_AST_H
