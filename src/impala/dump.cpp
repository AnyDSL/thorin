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

    std::ostream& print_block(const Stmt*);
    std::ostream& print_type(const Type*);

    Prec prec;
};

std::ostream& Printer::print_type(const Type* type) {
    if (type == nullptr) {
        return stream() << "<NULL>";
    } else if (type->isa<NoRet>()) {
        return stream() << "noret";
    } else if (type->isa<Void>()) {
        return stream() << "void";
    } else if (type->isa<TypeError>()) {
        return stream() << "<error>";
    } else if (auto tuple = type->isa<TupleType>()) {
        return dump_list([&](const Type* elem) { print_type(elem); }, tuple->elems(), "(", ")");
    } else if (auto fn = type->isa<FnType>()) {
        const Type* ret_type = fn->return_type();
        if (ret_type->isa<NoRet>())
            return dump_list([&](const Type* elem) { print_type(elem); }, fn->elems(), "fn(", ")");
        else
            return dump_list([&](const Type* elem) { print_type(elem); }, fn->elems().slice_front(fn->size()-1), "fn(", ") -> ") 
                << ret_type;
    } else if (auto generic = type->isa<Generic>()) {
        return stream() << Generic::to_string(generic->index());
    } else if (auto genref = type->isa<GenericRef>()) {
        return stream() << '<' << genref->fun()->symbol() << ", " << genref->generic() << '>';
    } else if (auto idtype = type->isa<IdType>()) {
        return stream() << idtype->name;
    } else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->kind()) {
#define IMPALA_TYPE(itype, atype) case Token::TYPE_##itype: return stream() << #itype;
#include "impala/tokenlist.h"
            default: ANYDSL2_UNREACHABLE;
        }
    }
    ANYDSL2_UNREACHABLE;
}

std::ostream& Printer::print_block(const Stmt* s) {
    if (s->isa<ScopeStmt>())
        s->print(*this);
    else {
        stream() << "{";
        up();
        s->print(*this);
        down();
        stream() << "}";
    }

    return stream();
}

//------------------------------------------------------------------------------

void ASTNode::dump() const { Printer p(std::cout, true); print(p) << std::endl; }
void Type::dump() const { Printer p(std::cout, true); p.print_type(this) << std::endl; }
std::ostream& GenericDecl::print(Printer& p) const { return p.stream() << symbol(); }

std::ostream& VarDecl::print(Printer& p) const { 
    p.stream() << (is_val() ? "val" : "var");
    if (orig_type())
        p.stream() << " : " << orig_type(); 
    return p.stream();
}

std::ostream& Scope::print(Printer& p) const {
    p.stream() << "{";
    p.up();
    if (!stmts_or_items().empty()) {
        for (auto i = stmts_or_items().cbegin(), e = stmts_or_items().cend() - 1; i != e; ++i) {
            (*i)->print(p);
            p.newline();
        }

        stmts_or_items().back()->print(p);
    }
    return p.down() << "}";
}

std::ostream& Proto::print(Printer& p) const {
    p.stream() << "extern " << symbol_ << " ";
    auto type = refined_fntype();
    return p.dump_list([&](const Type* type) { p.print_type(type); }, type->elems().slice_front(type->size()-1), "(", ")") 
            << " -> " << type->elems().back();
}

std::ostream& Fun::print(Printer& p) const {
    const VarDecl* ret_param = !params().empty() && params().back()->symbol() == Symbol("return") 
                             ? params().back() 
                             : nullptr;
    ArrayRef<const VarDecl*> params_ref = ret_param 
                                        ? ArrayRef<const VarDecl*>(&params().front(), params().size() - 1) 
                                        : params();

    p.dump_list([&](const VarDecl* decl) { decl->print(p); }, params_ref, is_lambda() ? "|" : "(", is_lambda() ? "|" : ")");

    if (ret_param)
        p.stream() << " -> " << orig_fntype()->return_type() << ' ';
    else
        p.stream() << ' ';

    return body()->print(p);
}

/*
 * Expr
 */

std::ostream& Literal::print(Printer& p) const {
    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: return p.stream() << (anydsl2::u64) box().get_##atype();
#include "impala/tokenlist.h"
        case LIT_bool: return p.stream() << (box().get_u1().get() ? "true" : "false");
        default: ANYDSL2_UNREACHABLE;
    }
}

std::ostream& Id::print(Printer& p) const { return p.stream() << symbol(); }
std::ostream& EmptyExpr::print(Printer& p) const { return p.stream() << "/*empty*/"; }
std::ostream& Tuple::print(Printer& p) const { return p.dump_list([&](const Expr* expr) { expr->print(p); }, ops(), "(", ")"); }

std::ostream& PrefixExpr::print(Printer& p) const {
    Prec r = PrecTable::prefix_r[kind()];
    Prec old = p.prec;

    const char* op;
    switch (kind()) {
#define IMPALA_PREFIX(tok, str, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
        default: ANYDSL2_UNREACHABLE;
    }

    p.stream() << op;
    p.prec = r;
    rhs()->print(p);
    p.prec = old;

    return p.stream();
}

std::ostream& InfixExpr::print(Printer& p) const {
    Prec l = PrecTable::infix_l[kind()];
    Prec r = PrecTable::infix_r[kind()];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p.stream() << "(";

    p.prec = l;
    lhs()->print(p);

    const char* op;
    switch (kind()) {
#define IMPALA_INFIX_ASGN(tok, str, lprec, rprec) case tok: op = str; break;
#define IMPALA_INFIX(     tok, str, lprec, rprec) case tok: op = str; break;
#include "impala/tokenlist.h"
    }

    p.stream() << " " << op << " ";

    p.prec = r;
    rhs()->print(p);
    p.prec = old;

    if (paren) p.stream() << ")";

    return p.stream();
}

std::ostream& PostfixExpr::print(Printer& p) const {
    Prec l = PrecTable::postfix_l[kind()];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p.stream() << "(";

    p.prec = l;
    lhs()->print(p);

    const char* op;
    switch (kind()) {
        case INC: op = "++"; break;
        case DEC: op = "--"; break;
        default: ANYDSL2_UNREACHABLE;
    }

    p.stream() << op;
    p.prec = old;

    if (paren) p.stream() << ")";

    return p.stream();
}

std::ostream& ConditionalExpr::print(Printer& p) const {
    Prec l = PrecTable::infix_l[Token::QUESTION_MARK];
    Prec r = PrecTable::infix_r[Token::QUESTION_MARK];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p.stream() << "(";

    p.prec = l;
    cond()->print(p) << " ? " << t_expr() << " : ";
    p.prec = r;
    f_expr()->print(p);
    p.prec = old;

    if (paren) p.stream() << ")";

    return p.stream();
}

std::ostream& IndexExpr::print(Printer& p) const {
    Prec l = PrecTable::postfix_l[Token::L_BRACKET];
    Prec old = p.prec;
    bool paren = !p.is_fancy() || p.prec > l;

    if (paren) p.stream() << "(";

    lhs()->print(p);
    p.stream() << "[";
    index()->print(p);
    p.stream() << "]";
    p.prec = old;

    if (paren) p.stream() << ")";

    return p.stream();
}

std::ostream& Call::print(Printer& p) const {
    assert(ops_.size() >= 1);
    ops_.front()->print(p);
    return p.dump_list([&](const Expr* expr) { expr->print(p); }, args(), "(", ")");
}

std::ostream& FunExpr::print(Printer& p) const { return fun()->print(p); }

/*
 * Stmt
 */

std::ostream& InitStmt::print(Printer& p) const {
    var_decl()->print(p);
    if (init()) {
        p.stream() << " = ";
        init()->print(p) << ";";
    }
    return p.stream();
}

std::ostream& ExprStmt::print(Printer& p) const {
    expr()->print(p);
    return p.stream() << ";";
}

std::ostream& IfElseStmt::print(Printer& p) const {
    p.stream() << "if (" << cond() << ") ";
    then_scope()->print(p);

    if (!else_scope()->empty()) {
        p.stream() << " else ";
        else_scope()->print(p);
    }

    return p.stream();
}

std::ostream& DoWhileStmt::print(Printer& p) const {
    p.stream() << "do ";
    body()->print(p);
    p.stream() << " while (";
    cond()->print(p);
    return p.stream() << ");";
}

std::ostream& ForStmt::print(Printer& p) const {
    if (is_while()) {
        p.stream() << "while (";
        cond()->print(p);
    } else {
        p.stream() << "for (";
        init()->print(p);
        p.stream() << " ";
        cond()->print(p);
        p.stream() << "; ";
        step()->print(p);
    }

    p.stream() << ") ";
    return body()->print(p);
}

std::ostream& ForeachStmt::print(Printer& p) const {
    p.stream() << "foreach ";
    call()->print(p) << ' ';
    return fun_expr()->print(p);
}

std::ostream& BreakStmt::print(Printer& p) const { return p.stream() << "break;"; }
std::ostream& ContinueStmt::print(Printer& p) const { return p.stream() << "continue;"; }
std::ostream& ScopeStmt::print(Printer& p) const { return scope()->print(p); }

std::ostream& ReturnStmt::print(Printer& p) const {
    p.stream() << "return";

    if (expr()) {
        p.stream() << " ";
        expr()->print(p);
    }

    return p.stream() << ";";
}

//------------------------------------------------------------------------------

std::ostream& FunItem::print(Printer& p) const {
    p.stream() << "fn " << fun()->symbol(); 
    if (!fun()->generics().empty())
        p.dump_list([&] (const GenericDecl* generic_decl) { generic_decl->print(p); }, fun()->generics(), "<", ">");

    return fun()->print(p);
}

std::ostream& TraitItem::print(Printer& p) const {
    assert( false && "todo" );
    return p.stream();
}

//------------------------------------------------------------------------------

void dump_prg(const Scope* scope, bool fancy, std::ostream& o) { 
    Printer p(o, fancy);
    for (auto stmt : scope->stmts_or_items()) {
        stmt->print(p);
        p.newline();
    }
}

void dump(const ASTNode* n, bool fancy, std::ostream& o) { Printer p(o, fancy); n->print(p); }
std::ostream& operator << (std::ostream& o, const ASTNode* n) { Printer p(o, true); return n->print(p); }
std::ostream& operator << (std::ostream& o, const Type* type) { return Printer(o, true).print_type(type); }

//------------------------------------------------------------------------------

} // namespace impala
