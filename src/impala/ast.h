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
}

namespace impala {

class CodeGen;
class VarDecl;
class Expr;
class NamedFct;
class Printer;
class ScopeStmt;
class Sema;
class Stmt;

typedef anydsl2::AutoVector<const VarDecl*> VarDecls;
typedef anydsl2::AutoVector<const Expr*> Exprs;
typedef std::vector<const NamedFct*> NamedFcts;
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

    const NamedFcts& named_fcts() const { return named_fcts_; }

private:

    anydsl2::AutoVector<const NamedFct*> named_fcts_;

    friend class Parser;
};

class Fct {
public:

    const ScopeStmt* body() const { return body_; }
    const VarDecl* param(size_t i) const { return params_[i]; }
    const VarDecls& params() const { return params_; }
    const anydsl2::Pi* pi() const { return pi_; }
    bool is_continuation() const;
    uintptr_t group() const { return group_; }
    anydsl2::Lambda* air_lambda() const { return air_lambda_; }
    const anydsl2::Param* ret_param() const { return ret_param_; }
    void fct_dump(Printer& p) const;
    void fct_check(Sema& sema) const;
    const anydsl2::Lambda* fct_emit(CodeGen& cg, anydsl2::Lambda* parent, const char* what) const;

private:

    void fct_set(const anydsl2::Pi* pi, const ScopeStmt* body) { 
        static uintptr_t group = 1; 
        group_ = group++; 
        pi_ = pi; 
        body_ = body; 
    }

    VarDecls params_;
    anydsl2::AutoPtr<const ScopeStmt> body_;
    uintptr_t group_;
    const anydsl2::Pi* pi_;
    mutable anydsl2::Lambda* air_lambda_;
    mutable const anydsl2::Param* ret_param_;

    friend class NamedFct;
    friend class FctExpr;
    friend class Parser;
    friend class CodeGen;
};

class Decl : public ASTNode {
protected:

    Decl() {}

public:

    Decl(const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2);

    anydsl2::Symbol symbol() const { return symbol_; }
    const anydsl2::Type* type() const { return type_; }
    void insert(Sema& sema) const;

protected:

    anydsl2::Symbol symbol_;
    const anydsl2::Type* type_;
};

class VarDecl : public Decl {
public:

    VarDecl(size_t handle, const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2)
        : Decl(tok, type, pos2)
        , handle_(handle)
    {}

    size_t handle() const { return handle_; }
    virtual void vdump(Printer& p) const;
    RefPtr emit(CodeGen& cg) const;

protected:

    size_t handle_;
};

class NamedFct : public Decl, public Fct {
public:

    NamedFct(bool ext)
        : extern_(ext)
    {}

    void set(const Token& tok, const anydsl2::Type* type, const anydsl2::Position& pos2);
    void check(Sema& sema) const;
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
    anydsl2::Array<const anydsl2::Def*> emit_ops(CodeGen& cg) const;
    virtual bool lvalue() const = 0;
    virtual RefPtr emit(CodeGen& cg) const = 0;

private:

    virtual const anydsl2::Type* vcheck(Sema& sema) const = 0;

protected:

    Exprs ops_;
    mutable const anydsl2::Type* type_;
};

class EmptyExpr : public Expr {
public:

    EmptyExpr(const anydsl2::Location& loc) { loc_ = loc; }

    virtual bool lvalue() const { return false; }
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

    Literal(const anydsl2::Location& loc, Kind kind, anydsl2::Box box);

    Kind kind() const { return kind_; }
    anydsl2::Box box() const { return box_; }

    virtual bool lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

    anydsl2::PrimTypeKind literal2type() const;

private:

    Kind kind_;
    anydsl2::Box box_;
};

class FctExpr : public Expr, public Fct {
public:

    virtual bool lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

    friend class Parser;
};

class Tuple : public Expr {
public:

    Tuple(const anydsl2::Position& pos1);

    virtual bool lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

    friend class Parser;
};

class Id : public Expr {
public:

    Id(const Token& tok);

    anydsl2::Symbol symbol() const { return symbol_; }
    const Decl* decl() const { return decl_; }

    virtual bool lvalue() const { return true; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

    anydsl2::Symbol symbol_;
    mutable const Decl* decl_; ///< Declaration of the variable in use.
};

class PrefixExpr : public Expr {
public:

    enum Kind {
#define IMPALA_PREFIX(tok, str, prec) tok = Token:: tok,
#include "impala/tokenlist.h"
    };

    PrefixExpr(const anydsl2::Position& pos1, Kind kind, const Expr* rhs);

    const Expr* rhs() const { return ops_[0]; }

    Kind kind() const { return kind_; }

    virtual bool lvalue() const { return true; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
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

    InfixExpr(const Expr* lhs, Kind kind, const Expr* rhs);

    const Expr* lhs() const { return ops_[0]; }
    const Expr* rhs() const { return ops_[1]; }

    Kind kind() const { return kind_; }

    virtual bool lvalue() const { return Token::is_asgn((TokenKind) kind()); }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
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

    PostfixExpr(const Expr* lhs, Kind kind, const anydsl2::Position& pos2);

    const Expr* lhs() const { return ops_[0]; }

    Kind kind() const { return kind_; }

    virtual bool lvalue() const { return false; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;

private:

    Kind kind_;
};

class IndexExpr : public Expr {
public:

    IndexExpr(const anydsl2::Position& pos1, const Expr* lhs, const Expr* index, const anydsl2::Position& pos2);

    const Expr* lhs() const { return ops_[0]; }
    const Expr* index() const { return ops_[1]; }

    virtual bool lvalue() const { return true; }
    virtual const anydsl2::Type* vcheck(Sema& sema) const;
    virtual RefPtr emit(CodeGen& cg) const;
    virtual void vdump(Printer& p) const;
};

class Call : public Expr {
public:

    Call(const Expr* fct);

    void append_arg(const Expr* expr) { ops_.push_back(expr); }
    void set_pos2(const anydsl2::Position& pos2);
    const Expr* to() const { return ops_.front(); }
    size_t num_args() const { return size() - 1; }
    const Expr* arg(size_t i) const { return op(i+1); }
    anydsl2::Location args_location() const;

    virtual bool lvalue() const { return false; }
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
    virtual void emit(CodeGen& cg) const = 0;
};

class ExprStmt : public Stmt {
public:

    ExprStmt(const Expr* expr, const anydsl2::Position& pos2);

    const Expr* expr() const { return expr_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl2::AutoPtr<const Expr> expr_;
};

class DeclStmt : public Stmt {
public:

    DeclStmt(const VarDecl* var_decl, const Expr* init, const anydsl2::Position& pos2);

    const VarDecl* var_decl() const { return var_decl_; }
    const Expr* init() const { return init_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl2::AutoPtr<const VarDecl> var_decl_;
    anydsl2::AutoPtr<const Expr> init_;
};

class IfElseStmt: public Stmt {
public:

    IfElseStmt(const anydsl2::Position& pos1, const Expr* cond, const Stmt* thenStmt, const Stmt* elseStmt);

    const Expr* cond() const { return cond_; }
    const Stmt* thenStmt() const { return thenStmt_; }
    const Stmt* elseStmt() const { return elseStmt_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

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

    void set(const Expr* cond, const Stmt* body) {
        cond_ = cond;
        body_ = body;
    }

private:

    anydsl2::AutoPtr<const Expr> cond_;
    anydsl2::AutoPtr<const Stmt> body_;
};

class DoWhileStmt : public Loop {
public:

    DoWhileStmt() {}

    void set(const anydsl2::Position& pos1, const Stmt* body, const Expr* cond, const anydsl2::Position& pos2);

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;
};

class ForStmt : public Loop {
public:

    ForStmt() {}

    void set(const anydsl2::Position& pos1, const Expr* cond, const Expr* step, const Stmt* body);
    void set(const DeclStmt* d) { init_decl_ = d; }
    void set(const ExprStmt* e) { init_expr_ = e; }
    void set_empty_init(const anydsl2::Position& pos);

    const DeclStmt* init_decl() const { return init_decl_; }
    const ExprStmt* init_expr() const { return init_expr_; }
    const Stmt* init() const { return (const Stmt*) ((uintptr_t) init_decl_.get() | (uintptr_t) init_expr_.get()); }
    const Expr* step() const { return step_; }
    bool is_while() const;

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl2::AutoPtr<const DeclStmt> init_decl_;
    anydsl2::AutoPtr<const ExprStmt> init_expr_;
    anydsl2::AutoPtr<const Expr> step_;
};

class BreakStmt : public Stmt {
public:

    BreakStmt(const anydsl2::Position& pos1, const anydsl2::Position& pos2, const Loop* loop);

    const Loop* loop() const { return loop_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;
private:

    const Loop* loop_;
};

class ContinueStmt : public Stmt {
public:

    ContinueStmt(const anydsl2::Position& pos1, const anydsl2::Position& pos2, const Loop* loop);

    const Loop* loop() const { return loop_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    const Loop* loop_;
};

class ReturnStmt : public Stmt {
public:

    ReturnStmt(const anydsl2::Position& pos1, const Expr* expr, const Fct* fct, const anydsl2::Position& pos2);

    const Expr* expr() const { return expr_; }
    const Fct* fct() const { return fct_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl2::AutoPtr<const Expr> expr_;
    const Fct* fct_;
};

class NamedFctStmt : public Stmt {
public:

    NamedFctStmt(const NamedFct* named_fct)
        : named_fct_(named_fct)
    {}

    const NamedFct* named_fct() const { return named_fct_; }

    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl2::AutoPtr<const NamedFct> named_fct_;
};

class ScopeStmt : public Stmt {
public:

    ScopeStmt() {}
    ScopeStmt(const anydsl2::Location& loc) {
        loc_ = loc;
    }

    const Stmts& stmts() const { return stmts_; }
    const NamedFcts& named_fcts() const { return named_fcts_; }
    void check_stmts(Sema& sema) const;

    virtual bool empty() const { return stmts_.empty(); }
    virtual void check(Sema& sema) const;
    virtual void vdump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    Stmts stmts_;
    NamedFcts named_fcts_;

    friend class Parser;
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_AST_H
