#include "anydsl2/util/for_all.h"

#include "anydsl2/printer.h"
#include "anydsl2/type.h"

#include "impala/ast.h"
#include "impala/dump.h"
#include "impala/prec.h"
#include "impala/type.h"

using anydsl2::Type;

namespace impala {

class Printer : public anydsl2::Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : anydsl2::Printer(o, fancy)
        , prec(BOTTOM)
    {}

    void dump_block(const Stmt* s);
    void dump(const ASTNode* n) { return n->dump(*this); }
    void dump(const anydsl2::Type* t);

    Prec prec;
};

void Printer::dump_block(const Stmt* s) {
    if (s->isa<ScopeStmt>())
        s->dump(*this);
    else {
        o << "{";
        up();
        s->dump(*this);
        down();
        o << "}";
    }
}

//------------------------------------------------------------------------------

void ASTNode::dump() const {
    ::impala::dump(this);
}

void Prg::dump(Printer& p) const {
    for_all (f, fcts()) {
        f->dump(p);
        p.newline();
    }
}

void Lambda::dump(Printer& p) const {
    if (!generics().empty()) {
        p << '<';
        ANYDSL2_DUMP_COMMA_LIST(p, generics());
        p << '>';
    }
    
    p << '(';
    ANYDSL2_DUMP_COMMA_LIST(p, params());
    p << ')';

    const Type* ret_type = return_type(pi());
    if (!ret_type->isa<NoRet>()) {
        p << " -> ";
        p.dump(ret_type);
        p << ' ';
    }

    p.dump_block(body());
}

void Fct::dump(Printer& p) const {
    p << "def " << decl()->symbol();
    lambda_.dump(p);
}

void Decl::dump(Printer& p) const {
    p << symbol() << " : ";
    p.dump(type());
}

/*
 * Expr
 */

void EmptyExpr::dump(Printer& p) const {
    p << "/*empty*/";
}

void Literal::dump(Printer& p) const {
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

void LambdaExpr::dump(Printer& p) const {
    p << "lambda";
    lambda_.dump(p);
}

void Tuple::dump(Printer& p) const {
    p << "#(";
    ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ")";
}

void Id::dump(Printer& p) const {
    p << symbol();
}

void PrefixExpr::dump(Printer& p) const {
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
    rhs()->dump(p);

    p.prec = old;
}

void InfixExpr::dump(Printer& p) const {
    Prec l = PrecTable::infix_l[kind()];
    Prec r = PrecTable::infix_r[kind()];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    p.prec = l;
    lhs()->dump(p);

    const char* op;
    switch (kind()) {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) case tok: op = str; break;
#define IMPALA_INFIX(     tok, str, lprec, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p << ' ' << op << ' ';

    p.prec = r;
    rhs()->dump(p);

    if (paren)
        p << ')';

    p.prec = old;
}

void PostfixExpr::dump(Printer& p) const {
    Prec l = PrecTable::postfix_l[kind()];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    p.prec = l;
    lhs()->dump(p);

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

void IndexExpr::dump(Printer& p) const {
    Prec l = PrecTable::postfix_l[Token::L_BRACKET];
    Prec old = p.prec;
    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p << '(';

    lhs()->dump(p);
    p << '[';
    index()->dump();
    p << ']';

    if (paren)
        p << ')';

    p.prec = old;
}

void Call::dump(Printer& p) const {
    assert(ops_.size() >= 1);

    ops_.front()->dump(p);
    p << '(';

    if (ops_.size() != 1) {
        for (Exprs::const_iterator i = ops_.begin() + 1, e = ops_.end() - 1; i != e; ++i) {
            (*i)->dump(p);
            p << ", ";
        }

        ops_.back()->dump(p);
    }

    p << ')';
}

/*
 * Stmt
 */

void DeclStmt::dump(Printer& p) const {
    decl()->dump(p);

    if (init()) {
        p << " = ";
        init()->dump(p);
    }

    p << ';';
}

void ExprStmt::dump(Printer& p) const {
    expr()->dump(p);
    p << ';';
}

void IfElseStmt::dump(Printer& p) const {
    p << "if (";
    cond()->dump(p);
    p << ") ";
    p.dump_block(thenStmt());

    if (!elseStmt()->empty()) {
        p << " else ";
        p.dump_block(elseStmt());
    }
}

void WhileStmt::dump(Printer& p) const {
    p << "while (";
    cond()->dump(p);
    p << ") ";

    p.dump_block(body());
}

void DoWhileStmt::dump(Printer& p) const {
    p << "do ";
    p.dump_block(body());
    p << " while (";
    cond()->dump(p);
    p << ");";
}

void ForStmt::dump(Printer& p) const {
    p << "for (";

    if (isDecl())
        initDecl()->dump(p);
    else
        initExpr()->dump(p);

    p << ' ';
    cond()->dump(p);
    p << "; ";

    step()->dump(p);
    p << ") ";

    p.dump_block(body());
}

void BreakStmt::dump(Printer& p) const {
    p << "break;";
}

void ContinueStmt::dump(Printer& p) const {
    p << "continue;";
}

void ReturnStmt::dump(Printer& p) const {
    p << "return";

    if (expr()) {
        p << ' ';
        expr()->dump(p);
    }

    p << ';';
}

void ScopeStmt::dump(Printer& p) const {
    p << "{";
    p.up();

    if (!stmts().empty()) {
        for (Stmts::const_iterator i = stmts().begin(), e = stmts().end() - 1; i != e; ++i) {
            (*i)->dump(p);
            p.newline();
        }

        stmts().back()->dump(p);
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
            ANYDSL2_DUMP_COMMA_LIST(*this, pi->elems().slice_front(pi->num_elems()-1));
            o << ") -> ";
            dump(ret_type);
        }
    } else if (const anydsl2::Generic* generic = t->isa<anydsl2::Generic>()) {
        o << generic->debug;
    } else if (t->isa<NoRet>()) {
        o << "noret";
    } else if (t->isa<TypeError>()) {
        o << "<type error>";
    } else {
        ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

void dump(const ASTNode* n, bool fancy /*= false*/, std::ostream& o /*= std::cout*/) {
    Printer p(o, fancy);
    n->dump(p);
}

void dump(const Type* t, bool fancy /*= false*/, std::ostream& o /*= std::cout*/) {
    Printer p(o, fancy);
    p.dump(t);
}

std::ostream& operator << (std::ostream& o, const ASTNode* n) {
    dump(n, true, o);
    return o;
}

std::ostream& operator << (std::ostream& o, const Type* t) {
    dump(t, true, o);
    return o;
}

//------------------------------------------------------------------------------

} // namespace impala
