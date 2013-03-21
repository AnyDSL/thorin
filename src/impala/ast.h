#ifndef IMPALA_AST_H
#define IMPALA_AST_H

#include <vector>

#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/assert.h"
#include "anydsl2/util/autoptr.h"
#include "anydsl2/util/box.h"
#include "anydsl2/util/cast.h"
#include "anydsl2/util/location.h"
#include "anydsl2/util/types.h"

#include "impala/token.h"

namespace anydsl2 {
    class Def;
    class Param;
    class Ref;
    class JumpTarget;
}

namespace impala {

class CodeGen;
class VarDecl;
class Expr;
class NamedFun;
class Printer;
class ScopeStmt;
class Sema;
class Stmt;

typedef anydsl2::AutoVector<const VarDecl*> VarDecls;
typedef anydsl2::AutoVector<const Expr*> Exprs;
typedef std::vector<const NamedFun*> NamedFuns;
typedef anydsl2::AutoVector<const Stmt*> Stmts;
typedef std::auto_ptr<const anydsl2::Ref> RefPtr;

class ASTNode : public anydsl2::HasLocation, public anydsl2::MagicCast {
public:

    virtual void vdump(Printer& p) const = 0;
    void dump() const;
};

class Prg : public ASTNode {
public:

    void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    void emit(CodeGen& cg) const;
    const anydsl2::AutoVector<const NamedFun*>& named_funs() const { return named_funs_; }

private:

    anydsl2::AutoVector<const NamedFun*> named_funs_;

    friend class Parser;
};

class Decl : public ASTNode {
protected:

    Decl() {}

public:

    Decl(const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2)
        : symbol_(tok.symbol())
        , type_(type)
    {
        set_loc(tok.pos1(), pos2);
    }

    anydsl2::Symbol symbol() const { return symbol_; }
    const anydsl2::Type* type() const { return type_; }
    void insert(Sema& sema) const;
    size_t depth() const { return depth_; }
    const Decl* shadows() const { return shadows_; }

protected:

    anydsl2::Symbol symbol_;
    const anydsl2::Type* type_;

private:

    mutable const Decl* shadows_;
    mutable size_t depth_;

    friend class Sema;
};

class Fun {
public:

    const ScopeStmt* body() const { return body_; }
    const VarDecl* param(size_t i) const { return params_[i]; }
    const VarDecls& params() const { return params_; }
    const anydsl2::Pi* pi() const { return pi_; }
    bool is_continuation() const;
    anydsl2::Lambda* lambda() const { return lambda_; }
    const anydsl2::Param* ret_param() const { return ret_param_; }
    void fun_dump(Printer& p) const;
    void fun_check(Sema& sema) const;
    const anydsl2::Lambda* emit_body(CodeGen& cg, anydsl2::Lambda* parent, const char* what) const;
    anydsl2::Lambda* emit_head(CodeGen& cg, anydsl2::Symbol symbol) const;

private:

    void fun_set(const anydsl2::Pi* pi, const ScopeStmt* body) { pi_ = pi; body_ = body; }

    VarDecls params_;
    anydsl2::AutoPtr<const ScopeStmt> body_;
    const anydsl2::Pi* pi_;
    mutable anydsl2::Lambda* lambda_;
    mutable const anydsl2::Param* ret_param_;

    friend class NamedFun;
    friend class FunExpr;
    friend class Parser;
    friend class CodeGen;
    friend class ForeachStmt;
};

class VarDecl : public Decl {
public:

    VarDecl(size_t handle, const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2)
        : Decl(tok, type, pos2)
        , handle_(handle)
        , is_address_taken_(false)
    {}

    size_t handle() const { return handle_; }
    bool is_address_taken() const { return is_address_taken_; }
    virtual void vdump(Printer& p) const;
    RefPtr emit(CodeGen& cg) const;

private:

    size_t handle_;
    mutable bool is_address_taken_;

    friend class Id;
};

class NamedFun : public Decl, public Fun {
public:

    NamedFun(bool ext)
        : extern_(ext)
    {}

    void set(const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2) {
        symbol_ = tok.symbol();
        type_ = type;
        set_loc(tok.pos1(), pos2);
    }

    void check(Sema& sema) const { return fun_check(sema); }
    virtual void vdump(Printer& p) const;
    void emit(CodeGen& cg) const;

private:

    void set(const Decl* decl, bool ext);

    bool extern_;

    friend class Parser;
};

//------------------------------------------------------------------------------

class Expr : public ASTNode {
public:

    Expr() 
        : type_(0) 
    {}

    const Exprs& ops() const { return ops_; }
    const Expr* op(size_t i) const { return ops_[i]; }
    size_t size() const { return ops_.size(); }
    bool empty() const { return size() == 0; }
    const anydsl2::Type* type() const { return type_; }
    const anydsl2::Type* check(Sema& sema) const { assert(!type_); return type_ = vcheck(sema); }
    anydsl2::Array<const anydsl2::Def*> emit_ops(CodeGen& cg, size_t additional_size = 0) const;
    virtual bool is_lvalue() const = 0;
    virtual RefPtr emit(CodeGen& cg) const = 0;
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;

private:

    virtual const anydsl2::Type* vcheck(Sema& sema) const = 0;

protected:

    Exprs ops_;
    mutable const anydsl2::Type* type_;
    
    friend class ForeachStmt;
};

class EmptyExpr : public Expr {
public:

    EmptyExpr(const anydsl2::Location& loc) { loc_ = loc; }

    virtual bool is_lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;
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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

    anydsl2::PrimTypeKind literal2type() const;

private:

    Kind kind_;
    anydsl2::Box box_;
};

class FunExpr : public Expr, public Fun {
public:

    virtual bool is_lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

    friend class Parser;
};

class Tuple : public Expr {
public:

    Tuple(const anydsl2::Position& pos1) { loc_.set_pos1(pos1); }

    virtual bool is_lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

    anydsl2::Symbol symbol_;
    mutable const Decl* decl_; ///< Declaration of the variable in use.
    
    friend class ForeachStmt;
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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;
    virtual void vdump(Printer& p) const;

private:

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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void emit_branch(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const;
    virtual void vdump(Printer& p) const;

private:

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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;
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
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;
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

    virtual bool is_lvalue() const { return false; }
    virtual RefPtr emit(CodeGen& cg) const;
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual void vdump(Printer& p) const;

    bool is_continuation_call() const;
};

//------------------------------------------------------------------------------

class Stmt : public ASTNode {
public:

    virtual bool empty() const { return false; }
    virtual void check(Sema& sema) const = 0;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const = 0;
};

class ExprStmt : public Stmt {
public:

    ExprStmt(const Expr* expr, const anydsl2::Position& pos2)
        : expr_(expr)
    {
        set_loc(expr->pos1(), pos2);
    }

    const Expr* expr() const { return expr_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const Expr> expr_;
};

class DeclStmt : public Stmt {
public:

    DeclStmt(const VarDecl* var_decl, const Expr* init, const anydsl2::Position& pos2)
        : var_decl_(var_decl)
        , init_(init)
    {
        set_loc(var_decl->pos1(), pos2);
    }

    const VarDecl* var_decl() const { return var_decl_; }
    const Expr* init() const { return init_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const VarDecl> var_decl_;
    anydsl2::AutoPtr<const Expr> init_;
};

class IfElseStmt: public Stmt {
public:

    IfElseStmt(const anydsl2::Position& pos1, const Expr* cond, const Stmt* thenStmt, const Stmt* elseStmt)
        : cond_(cond)
        , thenStmt_(thenStmt)
        , elseStmt_(elseStmt)
    {
        set_loc(pos1, elseStmt->pos2());
    }

    const Expr* cond() const { return cond_; }
    const Stmt* then_stmt() const { return thenStmt_; }
    const Stmt* else_stmt() const { return elseStmt_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const Expr> cond_;
    anydsl2::AutoPtr<const Stmt> thenStmt_;
    anydsl2::AutoPtr<const Stmt> elseStmt_;
};

class Loop : public Stmt {
public:

    Loop() {}

    const Expr* cond() const { return cond_; }
    const Stmt* body() const { return body_; }

protected:

    void set(const Expr* cond, const Stmt* body) { cond_ = cond; body_ = body; }

private:

    anydsl2::AutoPtr<const Expr> cond_;
    anydsl2::AutoPtr<const Stmt> body_;
};

class DoWhileStmt : public Loop {
public:

    DoWhileStmt() {}

    void set(const anydsl2::Position& pos1, const Stmt* body, const Expr* cond, const anydsl2::Position& pos2) {
        Loop::set(cond, body);
        set_loc(pos1, pos2);
    }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;
};

class ForStmt : public Loop {
public:

    ForStmt() {}

    void set(const anydsl2::Position& pos1, const Expr* cond, const Expr* step, const Stmt* body) {
        Loop::set(cond, body);
        step_ = step;
        set_loc(pos1, body->pos2());
    }
    void set(const DeclStmt* d) { init_decl_ = d; }
    void set(const ExprStmt* e) { init_expr_ = e; }
    void set_empty_init(const anydsl2::Position& pos) { set(new ExprStmt(new EmptyExpr(pos), pos)); }

    const DeclStmt* init_decl() const { return init_decl_; }
    const ExprStmt* init_expr() const { return init_expr_; }
    const Stmt* init() const { return (const Stmt*) ((uintptr_t) init_decl_.get() | (uintptr_t) init_expr_.get()); }
    const Expr* step() const { return step_; }
    bool is_while() const;

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const DeclStmt> init_decl_;
    anydsl2::AutoPtr<const ExprStmt> init_expr_;
    anydsl2::AutoPtr<const Expr> step_;
};


class ForeachStmt : public Stmt {
public:

    ForeachStmt() {}

    void set(size_t var_handle) { var_handle_ = var_handle; }
    void set(const anydsl2::Position& pos1, const Stmt* body) {
        body_ = body;
        set_loc(pos1, body->pos2());
    }
    void set(const VarDecl* d) { init_decl_ = d; }
    void set(const Expr* e) { init_expr_ = e; }

    // generator call
    const Expr* to() const { return ops_.front(); }
    size_t num_args() const { return size() - 1; }
    anydsl2::ArrayRef<const Expr*> args() const { return anydsl2::ArrayRef<const Expr*>(&*ops_.begin() + 1, num_args()); }
    const Expr* arg(size_t i) const { return op(i+1); }
    anydsl2::Location args_location() const;
    void append_arg(const Expr* expr) { ops_.push_back(expr); }
    const Exprs& ops() const { return ops_; }
    const Expr* op(size_t i) const { return ops_[i]; }
    size_t size() const { return ops_.size(); }

    const size_t var_handle() const { return var_handle_; }
    const Stmt* body() const { return body_; }
    const VarDecl* init_decl() const { return init_decl_; }
    const Expr* init_expr() const { return init_expr_; }
    const ASTNode* init() const { return (const ASTNode*) ((uintptr_t) init_decl_.get() | (uintptr_t) init_expr_.get()); }
    const anydsl2::Type* call_type() const { return call_type_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;
    virtual const anydsl2::Lambda* emit_ret(CodeGen& cg, anydsl2::Lambda* parent, anydsl2::JumpTarget& exit_bb) const;

private:

    anydsl2::AutoPtr<const Stmt> body_;
    anydsl2::AutoPtr<const VarDecl> init_decl_;
    anydsl2::AutoPtr<const Expr> init_expr_;
    Exprs ops_;
    size_t var_handle_;
    
    mutable const anydsl2::Type* left_type_;
    mutable const anydsl2::Type* inner_fun_type_;
    mutable const anydsl2::Pi* fun_type_;
    mutable const anydsl2::Type* call_type_;
};

class BreakStmt : public Stmt {
public:

    BreakStmt(const anydsl2::Position& pos1, const anydsl2::Position& pos2, const Loop* loop)
        : loop_(loop)
    {
        set_loc(pos1, pos2);
    }

    const Loop* loop() const { return loop_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;
private:

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

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

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

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const Expr> expr_;
    const Fun* fun_;
};

class NamedFunStmt : public Stmt {
public:

    NamedFunStmt(const NamedFun* named_fun)
        : named_fun_(named_fun)
    {}

    const NamedFun* named_fun() const { return named_fun_; }

    virtual void vdump(Printer& p) const;
    virtual void check(Sema& sema) const { named_fun()->check(sema); }
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    anydsl2::AutoPtr<const NamedFun> named_fun_;
};

class ScopeStmt : public Stmt {
public:

    ScopeStmt() {}
    ScopeStmt(const anydsl2::Location& loc) { loc_ = loc; }

    const Stmts& stmts() const { return stmts_; }
    const Stmt* stmt(size_t i) const { return stmts_[i]; }
    const NamedFuns& named_funs() const { return named_funs_; }
    void check_stmts(Sema& sema) const;

    virtual bool empty() const { return stmts_.empty(); }
    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg, anydsl2::JumpTarget& exit) const;

private:

    mutable Stmts stmts_;
    NamedFuns named_funs_;

    friend class Parser;
    friend class ForeachStmt;
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_AST_H
