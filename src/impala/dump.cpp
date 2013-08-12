#include "anydsl2/type.h"
#include "anydsl2/util/printer.h"

#include "impala/ast.h"
#include "impala/dump.h"
#include "impala/prec.h"
#include "impala/type.h"

using anydsl2::ArrayRef;
using anydsl2::Type;
using anydsl2::Symbol;

namespace impala {

class Printer : public anydsl2::Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : anydsl2::Printer(o, fancy)
        , prec(BOTTOM)
    {}

    Printer& print_block(const Stmt* s);
    Printer& operator << (const ASTNode* n) { return n->print(*this); }
    Printer& operator << (const Type* type);
    Printer& operator << (const char* str) { o << str; return *this; }
    Printer& operator << (Symbol sym) { o << sym; return *this; }

    Prec prec;
};

Printer& Printer::print_block(const Stmt* s) {
    if (s->isa<ScopeStmt>())
        s->print(*this);
    else {
        o << "{";
        up();
        s->print(*this);
        down();
        o << "}";
    }

    return *this;
}

//------------------------------------------------------------------------------

void ASTNode::dump() const {
    ::impala::dump(this);
}

Printer& Prg::print(Printer& p) const {
    for (auto global : globals()) {
        p.newline();
        global->print(p);
        p.newline();
    }

    return p;
}

Printer& Proto::print(Printer& p) const {
    p << "extern " << symbol_ << " ";
    ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, "(", pi()->elems().slice_front(pi()->size() - 1), ")");
    p << " -> " << pi()->elems().back();

    return p;
}

Printer& Fun::fun_print(Printer& p) const {
    // TODO generics
    const Type* ret_type = return_type(pi());
    ArrayRef<const VarDecl*> params_ref = 
        ret_type->isa<NoRet>() ? params() : ArrayRef<const VarDecl*>(&params().front(), params().size() - 1);

    ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, "(", params_ref, ")");

    if (!ret_type->isa<NoRet>())
        p << "-> " << ret_type << " ";

    return p.print_block(body());
}

Printer& NamedFun::print(Printer& p) const {
    p << "def " << symbol();
    return fun_print(p);
}

Printer& VarDecl::print(Printer& p) const {
    return p << symbol() << " : " << type();
}

/*
 * Expr
 */

Printer& EmptyExpr::print(Printer& p) const { return p << "/*empty*/"; }

Printer& Literal::print(Printer& p) const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: p.o << (anydsl2::u64) box().get_##atype(); return p;
#include "impala/tokenlist.h"
        case LIT_bool: return p << (box().get_u1().get() ? "true" : "false");
        default: ANYDSL2_UNREACHABLE;
    }
}

Printer& FunExpr::print(Printer& p) const {
    p << "lambda";
    return fun_print(p);
}

Printer& Tuple::print(Printer& p) const {
    ANYDSL2_DUMP_EMBRACING_COMMA_LIST(p, "#(", ops(), ")");
    return p;
}

Printer& Id::print(Printer& p) const {
    return p << symbol();
}

Printer& PrefixExpr::print(Printer& p) const {
    Prec r = PrecTable::prefix_r[kind()];
    Prec old = p.prec;

    const char* op;
    switch (kind()) {
#define IMPALA_PREFIX(tok, str, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
        default: ANYDSL2_UNREACHABLE;
    }

    p << op;
    p.prec = r;
    rhs()->print(p);
    p.prec = old;

    return p;
}

Printer& InfixExpr::print(Printer& p) const {
    Prec l = PrecTable::infix_l[kind()];
    Prec r = PrecTable::infix_r[kind()];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p << "(";

    p.prec = l;
    lhs()->print(p);

    const char* op;
    switch (kind()) {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) case tok: op = str; break;
#define IMPALA_INFIX(     tok, str, lprec, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p << " " << op << " ";

    p.prec = r;
    rhs()->print(p);
    p.prec = old;

    if (paren) p << ")";

    return p;
}

Printer& PostfixExpr::print(Printer& p) const {
    Prec l = PrecTable::postfix_l[kind()];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p << "(";

    p.prec = l;
    lhs()->print(p);

    const char* op;
    switch (kind()) {
        case INC: op = "++"; break;
        case DEC: op = "--"; break;
        default: ANYDSL2_UNREACHABLE;
    }

    p << op;
    p.prec = old;

    if (paren) p << ")";

    return p;
}

Printer& ConditionalExpr::print(Printer& p) const {
    Prec l = PrecTable::infix_l[Token::QUESTION_MARK];
    Prec r = PrecTable::infix_r[Token::QUESTION_MARK];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p << "(";

    p.prec = l;
    cond()->print(p) << " ? " << t_expr() << " : ";
    p.prec = r;
    f_expr()->print(p);
    p.prec = old;

    if (paren) p << ")";

    return p;
}

Printer& IndexExpr::print(Printer& p) const {
    Prec l = PrecTable::postfix_l[Token::L_BRACKET];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p << "(";

    lhs()->print(p);
    p << "[";
    index()->print(p);
    p << "]";
    p.prec = old;

    if (paren) p << ")";

    return p;
}

Printer& Call::print(Printer& p) const {
    assert(ops_.size() >= 1);

    ops_.front()->print(p);
    p << "(";

    if (ops_.size() != 1) {
        for (auto i = ops_.cbegin() + 1, e = ops_.cend() - 1; i != e; ++i) {
            (*i)->print(p);
            p << ", ";
        }

        ops_.back()->print(p);
    }

    return p << ")";
}

/*
 * Stmt
 */

Printer& DeclStmt::print(Printer& p) const {
    var_decl()->print(p);

    if (init()) {
        p << " = ";
        init()->print(p);
    }

    return p << ";";
}

Printer& ExprStmt::print(Printer& p) const {
    expr()->print(p);
    return p << ";";
}

Printer& IfElseStmt::print(Printer& p) const {
    p << "if (" << cond() << ") ";
    p.print_block(then_stmt());

    if (!else_stmt()->empty()) {
        p << " else ";
        p.print_block(else_stmt());
    }

    return p;
}

Printer& DoWhileStmt::print(Printer& p) const {
    p << "do ";
    p.print_block(body());
    p << " while (";
    cond()->print(p);
    p << ");";

    return p;
}

Printer& ForStmt::print(Printer& p) const {
    if (is_while()) {
        p << "while (";
        cond()->print(p);
    } else {
        p << "for (";
        init()->print(p);
        p << " ";
        cond()->print(p);
        p << "; ";
        step()->print(p);
    }
    p << ") ";
    p.print_block(body());

    return p;
}

Printer& ForeachStmt::print(Printer& p) const {
    p << "foreach (";
    init()->print(p);
    p << " <- ";
    call()->print(p);
    p << ")";
    p.print_block(body());

    return p;
}

Printer& BreakStmt::print(Printer& p) const { p << "break;"; return p; }
Printer& ContinueStmt::print(Printer& p) const { p << "continue;"; return p; }

Printer& ReturnStmt::print(Printer& p) const {
    p << "return";

    if (expr()) {
        p << " ";
        expr()->print(p);
    }

    p << ";";

    return p;
}

Printer& NamedFunStmt::print(Printer& p) const { named_fun()->print(p); return p; }

Printer& ScopeStmt::print(Printer& p) const {
    p << "{";
    p.up();

    if (!stmts().empty()) {
        for (auto i = stmts().cbegin(), e = stmts().cend() - 1; i != e; ++i) {
            (*i)->print(p);
            p.newline();
        }

        stmts().back()->print(p);
    }
    p.down();
    p << "}";

    return p;
}

//------------------------------------------------------------------------------

Printer& Printer::operator << (const anydsl2::Type* t) { 
    if (t->isa<anydsl2::PrimType>()) {
        t->print(*this);
    } else if (const anydsl2::Sigma* sigma = t->isa<anydsl2::Sigma>()) {
        ANYDSL2_DUMP_EMBRACING_COMMA_LIST(*this, "#(", sigma->elems(), ")");
    } else if (const anydsl2::Pi* pi = t->isa<anydsl2::Pi>()) {
        const Type* ret_type = return_type(pi);
        if (ret_type->isa<NoRet>()) {
            ANYDSL2_DUMP_EMBRACING_COMMA_LIST(*this, "pi(", pi->elems(), ")");
        } else {
            ANYDSL2_DUMP_EMBRACING_COMMA_LIST(*this, "pi(", pi->elems().slice_front(pi->size()-1), ") -> ");
            dump(ret_type);
        }
    } else if (t->isa<Void>()) {
      o << "void";
    } else if (t->isa<NoRet>()) {
      o << "noret";
    } else if (t->isa<TypeError>()) {
      o << "<type error>";
    } else {
        t->print(*this);
    }

    return *this;
}

//------------------------------------------------------------------------------

void dump(const ASTNode* n, bool fancy, std::ostream& o) {
    Printer p(o, fancy);
    n->print(p);
}

void dump(const Type* t, bool fancy, std::ostream& o) {
    Printer(o, fancy) << t;
}

std::ostream& operator << (std::ostream& o, const ASTNode* n) {
    dump(n, true, o);
    return o;
}

//------------------------------------------------------------------------------

} // namespace impala
