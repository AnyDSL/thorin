#ifndef IMPALA_AST_H
#define IMPALA_AST_H

#include <vector>

#include "anydsl/util/array.h"
#include "anydsl/util/assert.h"
#include "anydsl/util/autoptr.h"
#include "anydsl/util/box.h"
#include "anydsl/util/cast.h"
#include "anydsl/util/location.h"

#include "impala/token.h"

namespace anydsl {
    class Def;
    class Var;
}

namespace impala {

class CodeGen;
class Decl;
class Expr;
class Fct;
class Pi;
class Printer;
class ScopeStmt;
class Sema;
class Stmt;
class Type;

typedef anydsl::AutoVector<const Decl*> Decls;
typedef anydsl::AutoVector<const Expr*> Exprs;
typedef anydsl::AutoVector<const Fct*>  Fcts;
typedef anydsl::AutoVector<const Stmt*> Stmts;

class ASTNode : public anydsl::HasLocation, public anydsl::MagicCast {
public:

    virtual void check(Sema& sema) const = 0;
    virtual void dump(Printer& p) const = 0;

    void dump() const;
};

class Prg : public ASTNode {
public:

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    void emit(CodeGen& cg) const;

    const Fcts& fcts() const { return fcts_; }

private:

    Fcts fcts_;

    friend class Parser;
};

class Fct : public ASTNode {
public:

    Fct() {}

    const Decl* decl() const { return decl_; }
    anydsl::Symbol symbol() const;
    const ScopeStmt* body() const { return body_; }
    const Decls& params() const { return params_; }
    const Pi* pi() const;
    bool continuation() const;

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    void emit(CodeGen& cg) const;

private:

    void set(const Decl* decl, const ScopeStmt* body);

    anydsl::AutoPtr<const Decl> decl_;
    Decls params_;
    anydsl::AutoPtr<const ScopeStmt> body_;

    friend class Parser;
};

class Decl : public ASTNode {
public:

    Decl(const Token& tok, const Type* type, const anydsl::Position& pos2);

    anydsl::Symbol symbol() const { return symbol_; }
    const Type* type() const { return type_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    anydsl::Var* emit(CodeGen& cg) const;

private:

    anydsl::Symbol symbol_;
    const Type* type_;
};

//------------------------------------------------------------------------------

class Expr : public ASTNode {
public:

    bool lvalue() const { return lvalue_; }
    const Type* type() const { return type_; }
    virtual anydsl::Var* lemit(CodeGen& cg) const { ANYDSL_UNREACHABLE; }
    virtual const anydsl::Def* remit(CodeGen& cg) const = 0;
    const Exprs& ops() const { return ops_; }

protected:

    Exprs ops_;

    mutable bool lvalue_;
    mutable const Type* type_;
};

class EmptyExpr : public Expr {
public:

    EmptyExpr(const anydsl::Location& loc) {
        loc_ = loc;
    }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
};

class Literal : public Expr {
public:

    enum Kind {
#define IMPALA_LIT(itype, atype) LIT_##itype = Token::LIT_##itype,
#include "impala/tokenlist.h"
        LIT_bool
    };

    Literal(const anydsl::Location& loc, Kind kind, anydsl::Box box);

    Kind kind() const { return kind_; }
    anydsl::Box box() const { return box_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;

private:

    Kind kind_;
    anydsl::Box box_;
};

class Id : public Expr {
public:

    Id(const Token& tok);

    anydsl::Symbol symbol() const { return symbol_; }
    const Decl* decl() const { return decl_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
    virtual anydsl::Var* lemit(CodeGen& cg) const;

private:

    anydsl::Symbol symbol_;
    mutable const Decl* decl_; ///< Declaration of the variable in use.
};

class PrefixExpr : public Expr {
public:

    enum Kind {
#define IMPALA_PREFIX(tok, str, prec) tok = Token:: tok,
#include "impala/tokenlist.h"
    };

    PrefixExpr(const anydsl::Position& pos1, Kind kind, const Expr* rhs);

    const Expr* rhs() const { return ops_[0]; }

    Kind kind() const { return kind_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
    virtual anydsl::Var* lemit(CodeGen& cg) const;

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

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
    virtual anydsl::Var* lemit(CodeGen& cg) const;

private:

    Kind kind_;
};

/**
 * Just for expr++ and expr--.
 * For indexing and function calls use IndexExpr or Call, respectively.
 */
class PostfixExpr : public Expr {
public:

    enum Kind {
        INC = Token::INC,
        DEC = Token::DEC
    };

    PostfixExpr(const Expr* lhs, Kind kind, const anydsl::Position& pos2);

    const Expr* lhs() const { return ops_[0]; }

    Kind kind() const { return kind_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
    virtual anydsl::Var* lemit(CodeGen& cg) const;

private:

    Kind kind_;
};

class Call : public Expr {
public:

    Call(const Expr* fct);

    void append_arg(const Expr* expr) { ops_.push_back(expr); }
    void setLoc(const anydsl::Position& pos2);
    anydsl::Array<const anydsl::Def*> emit_ops(CodeGen& cg) const;

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual const anydsl::Def* remit(CodeGen& cg) const;
};

//------------------------------------------------------------------------------

class Stmt : public ASTNode {
public:

    virtual bool empty() const { return false; }
    virtual void emit(CodeGen& cg) const = 0;
};

class ExprStmt : public Stmt {
public:

    ExprStmt(const Expr* expr, const anydsl::Position& pos2);

    const Expr* expr() const { return expr_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl::AutoPtr<const Expr> expr_;
};

class DeclStmt : public Stmt {
public:

    DeclStmt(const Decl* decl, const Expr* init, const anydsl::Position& pos2);

    const Decl* decl() const { return decl_; }
    const Expr* init() const { return init_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl::AutoPtr<const Decl> decl_;
    anydsl::AutoPtr<const Expr> init_;
};

class IfElseStmt: public Stmt {
public:

    IfElseStmt(const anydsl::Position& pos1, const Expr* cond, const Stmt* thenStmt, const Stmt* elseStmt);

    const Expr* cond() const { return cond_; }
    const Stmt* thenStmt() const { return thenStmt_; }
    const Stmt* elseStmt() const { return elseStmt_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl::AutoPtr<const Expr> cond_;
    anydsl::AutoPtr<const Stmt> thenStmt_;
    anydsl::AutoPtr<const Stmt> elseStmt_;
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

    anydsl::AutoPtr<const Expr> cond_;
    anydsl::AutoPtr<const Stmt> body_;
};

class WhileStmt : public Loop {
public:

    WhileStmt() {}
        
    void set(const anydsl::Position& pos1, const Expr* cond, const Stmt* body);

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;
};

class DoWhileStmt : public Loop {
public:

    DoWhileStmt() {}

    void set(const anydsl::Position& pos1, const Stmt* body, const Expr* cond, const anydsl::Position& pos2);

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;
};

class ForStmt : public Loop {
public:

    ForStmt() {}
    virtual ~ForStmt();

    void set(const anydsl::Position& pos1, const Expr* cond, const Expr* step, const Stmt* body);
    void set(const DeclStmt* d) { initDecl_ = d; isDecl_ = true; }
    void set(const ExprStmt* e) { initExpr_ = e; isDecl_ = false; }

    const DeclStmt* initDecl() const { return initDecl_; }
    const ExprStmt* initExpr() const { return initExpr_; }
    const Stmt* init() const { return isDecl_ ? (const Stmt*) initDecl_ : (const Stmt*) initExpr_; }
    const Expr* step() const { return step_; }

    bool isDecl() const { return isDecl_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    union {
        const DeclStmt* initDecl_;
        const ExprStmt* initExpr_;
    };

    anydsl::AutoPtr<const Expr> step_;
    bool isDecl_;
};

class BreakStmt : public Stmt {
public:

    BreakStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop);

    const Loop* loop() const { return loop_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;
private:

    const Loop* loop_;
};

class ContinueStmt : public Stmt {
public:

    ContinueStmt(const anydsl::Position& pos1, const anydsl::Position& pos2, const Loop* loop);

    const Loop* loop() const { return loop_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    const Loop* loop_;
};

class ReturnStmt : public Stmt {
public:

    ReturnStmt(const anydsl::Position& pos1, const Expr* expr, const Fct* fct, const anydsl::Position& pos2);

    const Expr* expr() const { return expr_; }
    const Fct* fct() const { return fct_; }

    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    anydsl::AutoPtr<const Expr> expr_;
    const Fct* fct_;
};

class ScopeStmt : public Stmt {
public:

    ScopeStmt() {}
    ScopeStmt(const anydsl::Location& loc) {
        loc_ = loc;
    }

    const Stmts& stmts() const { return stmts_; }

    virtual bool empty() const { return stmts_.empty(); }
    virtual void check(Sema& sema) const;
    virtual void dump(Printer& p) const;
    virtual void emit(CodeGen& cg) const;

private:

    Stmts stmts_;

    friend class Parser;
};

//------------------------------------------------------------------------------

} // namespace impala

#endif // IMPALA_AST_H
