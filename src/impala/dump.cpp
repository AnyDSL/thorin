#include "impala/ast.h"

#include "anydsl/util/for_all.h"

#include "impala/prec.h"
#include "impala/type.h"

namespace impala {

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , prec(BOTTOM)
        , fancy_(fancy)
        , indent_(0)
    {}

    void newline();
    void up();
    void down();

    bool fancy() const { return fancy_; }

    std::ostream& o;
    Prec prec;

    void dumpBlock(const Stmt* s);

private:

    bool fancy_;
    int indent_;
};

void Printer::newline() {
    o << '\n';
    for (int i = 0; i < indent_; ++i)
        o << "    ";
}

void Printer::up() {
    ++indent_;
    newline();
}

void Printer::down() {
    --indent_;
    newline();
}

//------------------------------------------------------------------------------

void dump(const ASTNode* n, bool fancy /*= false*/, std::ostream& o /*= std::cout*/) {
    Printer p(o, fancy);
    n->dump(p);
}

void dump(const Type* t, bool fancy /*= false*/, std::ostream& o /*= std::cout*/) {
    Printer p(o, fancy);
    t->dump(p);
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

void Printer::dumpBlock(const Stmt* s) {
    if (s->isa<ScopeStmt>())
        s->dump(*this);
    else {
        //o << "{";
        up();
        s->dump(*this);
        --indent_;
        //down();
        //o << "}";
    }
}

//------------------------------------------------------------------------------

void Prg::dump(Printer& p) const {
    for_all (f, fcts()) {
        f->dump(p);
        p.newline();
    }
}

void Fct::dump(Printer& p) const {
    p.o << "def " << symbol() << '(';

    if (!params().empty()) {
        for (Decls::const_iterator i = params().begin(), e = params().end() - 1; i != e; ++i) {
            (*i)->dump(p);
            p.o << ", ";
        }

        params().back()->dump(p);
    }

    p.o << ')';

    if (retType()) {
        p.o << " -> ";
        retType()->dump(p);
        p.o << ' ';
    }
    p.dumpBlock(body());
}

void Decl::dump(Printer& p) const {
    p.o << symbol() << " : ";
    type()->dump(p);
}

/*
 * Expr
 */

void EmptyExpr::dump(Printer& p) const {
    p.o << "/*empty*/";
}

void Literal::dump(Printer& p) const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: { \
            p.o << (anydsl::u64) value().get_##atype(); \
            return; \
        }
#include "impala/tokenlist.h"
        case LIT_bool:
            if (value().bool_) 
                p.o << "true";
            else
                p.o << "false";
            return;
    }
}

void Id::dump(Printer& p) const {
    p.o << symbol();
}

void PrefixExpr::dump(Printer& p) const {
    Prec r = PrecTable::prefix_r[kind()];
    Prec old = p.prec;

    const char* op;
    switch (kind()) {
#define IMPALA_PREFIX(tok, str, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p.o << op;

    p.prec = r;
    rexpr()->dump(p);

    p.prec = old;
}

void InfixExpr::dump(Printer& p) const {
    Prec l = PrecTable::infix_l[kind()];
    Prec r = PrecTable::infix_r[kind()];
    Prec old = p.prec;

    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p.o << '(';

    p.prec = l;
    lexpr()->dump(p);

    const char* op;
    switch (kind()) {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) case tok: op = str; break;
#define IMPALA_INFIX(     tok, str, lprec, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p.o << ' ' << op << ' ';

    p.prec = r;
    rexpr()->dump(p);

    if (paren)
        p.o << ')';

    p.prec = old;
}

void PostfixExpr::dump(Printer& p) const {
    Prec l = PrecTable::postfix_l[kind()];
    Prec old = p.prec;

    bool paren = !p.fancy() || p.prec > l;

    if (paren)
        p.o << '(';

    p.prec = l;
    lexpr()->dump(p);

    const char* op;
    switch (kind()) {
        case INC: op = "++"; break;
        case DEC: op = "--"; break;
    }

    p.o << op;

    if (paren)
        p.o << ')';

    p.prec = old;
}

/*
 * Stmt
 */

void DeclStmt::dump(Printer& p) const {
    decl()->dump(p);

    if (init()) {
        p.o << " = ";
        init()->dump(p);
    }

    p.o << ';';
}

void ExprStmt::dump(Printer& p) const {
    expr()->dump(p);
    p.o << ';';
}

void IfElseStmt::dump(Printer& p) const {
    p.o << "if (";
    cond()->dump(p);
    p.o << ") ";
    p.dumpBlock(thenStmt());

    if (!elseStmt()->isEmpty()) {
        p.o << " else ";
        p.dumpBlock(elseStmt());
    }
}

void WhileStmt::dump(Printer& p) const {
    p.o << "while (";
    cond()->dump(p);
    p.o << ") ";

    p.dumpBlock(body());
}

void DoWhileStmt::dump(Printer& p) const {
    p.o << "do ";
    p.dumpBlock(body());
    p.o << " while (";
    cond()->dump(p);
    p.o << ");";
}

void ForStmt::dump(Printer& p) const {
    p.o << "for (";

    if (isDecl())
        initDecl()->dump(p);
    else
        initExpr()->dump(p);

    p.o << ' ';
    cond()->dump(p);
    p.o << "; ";

    inc()->dump(p);
    p.o << ") ";

    p.dumpBlock(body());
}

void BreakStmt::dump(Printer& p) const {
    p.o << "break;";
}

void ContinueStmt::dump(Printer& p) const {
    p.o << "continue;";
}

void ReturnStmt::dump(Printer& p) const {
    p.o << "return";

    if (expr()) {
        p.o << ' ';
        expr()->dump(p);
    }

    p.o << ';';
}

void ScopeStmt::dump(Printer& p) const {
    p.o << "{";
    p.up();

    if (!stmts().empty()) {
        for (Stmts::const_iterator i = stmts().begin(), e = stmts().end() - 1; i != e; ++i) {
            (*i)->dump(p);
            p.newline();
        }

        stmts().back()->dump(p);
    }
    p.down();
    p.o << "}";
}

//------------------------------------------------------------------------------

/*
 * Type
 */

void PrimType::dump(Printer& p) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) case TYPE_##itype: p.o << #itype; return;
#include "impala/tokenlist.h"
    }
}

void Void::dump(Printer& p) const {
    p.o << "void";
}

void TypeError::dump(Printer& p) const {
    p.o << "<type error>";
}

} // namespace impala
