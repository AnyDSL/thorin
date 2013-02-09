#include "anydsl2/util/for_all.h"

#include "anydsl2/printer.h"
#include "anydsl2/type.h"

#include "impala/ast.h"
#include "impala/dump.h"
#include "impala/prec.h"
#include "impala/type.h"

using anydsl2::ArrayRef;
using anydsl2::Type;

namespace impala {

class Printer : public anydsl2::Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : anydsl2::Printer(o, fancy)
        , prec(BOTTOM)
    {}

    void dump_block(const Stmt* s);
    void dump(const ASTNode* n) { return n->vdump(*this); }
    void dump(const anydsl2::Type* t);

    Prec prec;
};

void Printer::dump_block(const Stmt* s) {
    if (s->isa<ScopeStmt>())
        s->vdump(*this);
    else {
        o << "{";
        up();
        s->vdump(*this);
        down();
        o << "}";
    }
}

//------------------------------------------------------------------------------

void ASTNode::dump() const {
    ::impala::dump(this);
}

void Prg::vdump(Printer& p) const {
    for_all (f, named_funs()) {
        f->vdump(p);
        p.newline();
    }
}

void Fun::fun_dump(Printer& p) const {
#if 0
    if (!generics().empty()) {
        p << "[[";
        ANYDSL2_DUMP_COMMA_LIST(p, generics());
        p << "]]";
    }
#endif
    
    const Type* ret_type = return_type(pi());
    ArrayRef<const VarDecl*> params_ref = 
        ret_type->isa<NoRet>() ? params() : ArrayRef<const VarDecl*>(&params().front(), params().size() - 1);

    p << '(';
    ANYDSL2_DUMP_COMMA_LIST(p, params_ref);
    p << ") ";

    if (!ret_type->isa<NoRet>()) {
        p << "-> ";
        p.dump(ret_type);
        p << ' ';
    }

    p.dump_block(body());
}

void NamedFun::vdump(Printer& p) const {
    p << "def " << symbol();
    fun_dump(p);
}

void VarDecl::vdump(Printer& p) const {
    p << symbol() << " : ";
    p.dump(type());
}

/*
 * Expr
 */

void EmptyExpr::vdump(Printer& p) const {
    p << "/*empty*/";
}

void Literal::vdump(Printer& p) const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: { \
            p << (anydsl2::u64) box().get_##atype(); \
            return; \
        }
#include "impala/tokenlist.h"
        case LIT_bool:
            if (box().get_u1().get()) 
                p << "true";
            else
                p << "false";
            return;
    }
}

void FunExpr::vdump(Printer& p) const {
    p << "lambda";
    fun_dump(p);
}

void Tuple::vdump(Printer& p) const {
    p << "#(";
    ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ")";
}

void Id::vdump(Printer& p) const {
    p << symbol();
}

void PrefixExpr::vdump(Printer& p) const {
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
    rhs()->vdump(p);

    p.prec = old;
}

void InfixExpr::vdump(Printer& p) const {
    Prec l = PrecTable::infix_l[kind()];
    Prec r = PrecTable::infix_r[kind()];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    p.prec = l;
    lhs()->vdump(p);

    const char* op;
    switch (kind()) {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) case tok: op = str; break;
#define IMPALA_INFIX(     tok, str, lprec, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p << ' ' << op << ' ';

    p.prec = r;
    rhs()->vdump(p);

    if (paren)
        p << ')';

    p.prec = old;
}

void PostfixExpr::vdump(Printer& p) const {
    Prec l = PrecTable::postfix_l[kind()];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    p.prec = l;
    lhs()->vdump(p);

    const char* op;
    switch (kind()) {
        case INC: op = "++"; break;
        case DEC: op = "--"; break;
        default: ANYDSL2_UNREACHABLE;
    }

    p << op;

    if (paren)
        p << ')';

    p.prec = old;
}

void ConditionalExpr::vdump(Printer& p) const {
    Prec l = PrecTable::infix_l[Token::QUESTION_MARK];
    Prec r = PrecTable::infix_r[Token::QUESTION_MARK];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    p.prec = l;
    cond()->vdump(p);
    p << " ? ";
    t_expr()->vdump(p);
    p << " : ";
    p.prec = r;
    f_expr()->vdump(p);

    if (paren)
        p << ')';

    p.prec = old;
}

void IndexExpr::vdump(Printer& p) const {
    Prec l = PrecTable::postfix_l[Token::L_BRACKET];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    lhs()->vdump(p);
    p << '[';
    index()->vdump(p);
    p << ']';

    if (paren)
        p << ')';

    p.prec = old;
}

void Call::vdump(Printer& p) const {
    assert(ops_.size() >= 1);

    ops_.front()->vdump(p);
    p << '(';

    if (ops_.size() != 1) {
        for (Exprs::const_iterator i = ops_.begin() + 1, e = ops_.end() - 1; i != e; ++i) {
            (*i)->vdump(p);
            p << ", ";
        }

        ops_.back()->vdump(p);
    }

    p << ')';
}

/*
 * Stmt
 */

void DeclStmt::vdump(Printer& p) const {
    var_decl()->vdump(p);

    if (init()) {
        p << " = ";
        init()->vdump(p);
    }

    p << ';';
}

void ExprStmt::vdump(Printer& p) const {
    expr()->vdump(p);
    p << ';';
}

void IfElseStmt::vdump(Printer& p) const {
    p << "if (";
    cond()->vdump(p);
    p << ") ";
    p.dump_block(then_stmt());

    if (!else_stmt()->empty()) {
        p << " else ";
        p.dump_block(else_stmt());
    }
}

void DoWhileStmt::vdump(Printer& p) const {
    p << "do ";
    p.dump_block(body());
    p << " while (";
    cond()->vdump(p);
    p << ");";
}

void ForStmt::vdump(Printer& p) const {
    if (is_while()) {
        p << "while (";
        cond()->vdump(p);
    } else {
        p << "for (";
        init()->vdump(p);
        p << ' ';
        cond()->vdump(p);
        p << "; ";
        step()->vdump(p);
    }
    p << ") ";
    p.dump_block(body());
}

void BreakStmt::vdump(Printer& p) const {
    p << "break;";
}

void ContinueStmt::vdump(Printer& p) const {
    p << "continue;";
}

void ReturnStmt::vdump(Printer& p) const {
    p << "return";

    if (expr()) {
        p << ' ';
        expr()->vdump(p);
    }

    p << ';';
}

void NamedFunStmt::vdump(Printer& p) const { named_fun()->vdump(p); }

void ScopeStmt::vdump(Printer& p) const {
    p << "{";
    p.up();

    if (!stmts().empty()) {
        for (Stmts::const_iterator i = stmts().begin(), e = stmts().end() - 1; i != e; ++i) {
            (*i)->vdump(p);
            p.newline();
        }

        stmts().back()->vdump(p);
    }
    p.down();
    p << "}";
}

//------------------------------------------------------------------------------

void Printer::dump(const anydsl2::Type* t) { 
    if (t->isa<anydsl2::PrimType>()) {
        t->vdump(*this);
    } else if (const anydsl2::Sigma* sigma = t->isa<anydsl2::Sigma>()) {
        o << "#(";
        ANYDSL2_DUMP_COMMA_LIST(*this, sigma->elems());
        o << ")";
    } else if (const anydsl2::Pi* pi = t->isa<anydsl2::Pi>()) {
        o << "pi(";

        const Type* ret_type = return_type(pi);
        if (ret_type->isa<NoRet>()) {
            ANYDSL2_DUMP_COMMA_LIST(*this, pi->elems());
            o << ")";
        } else {
            ANYDSL2_DUMP_COMMA_LIST(*this, pi->elems().slice_front(pi->size()-1));
            o << ") -> ";
            dump(ret_type);
        }
#if 0
    } else if (const anydsl2::Opaque* opaque = t->isa<anydsl2::Opaque>()) {
        opaque->vdump(*this);
#endif
    } else if (const anydsl2::Generic* generic = t->isa<anydsl2::Generic>()) {
        generic->vdump(*this);
    } else if (t->isa<Void>()) {
        o << "void";
    } else if (t->isa<NoRet>()) {
        o << "noret";
    } else if (t->isa<TypeError>()) {
        o << "<type error>";
    } else {
        ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

void dump(const ASTNode* n, bool fancy, std::ostream& o) {
    Printer p(o, fancy);
    n->vdump(p);
}

void dump(const Type* t, bool fancy, std::ostream& o) {
    Printer p(o, fancy);
    p.dump(t);
}

std::ostream& operator << (std::ostream& o, const ASTNode* n) {
    dump(n, true, o);
    return o;
}

//------------------------------------------------------------------------------

} // namespace impala
