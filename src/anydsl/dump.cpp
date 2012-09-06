#include <boost/typeof/typeof.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/for_all.h"
#include "anydsl/printer.h"

#define ANYDSL_DUMP_COMMA_LIST(p, list) \
    const BOOST_TYPEOF((list))& l = (list); \
    if (!l.empty()) { \
        boost::remove_const<BOOST_TYPEOF(l)>::type::const_iterator i = l.begin(), e = l.end() - 1; \
        for (; i != e; ++i) { \
            (p).dump(*i); \
            (p) << ", "; \
        } \
        (p).dump(*i); \
    }

namespace anydsl {

// Literal

void Any::vdump(Printer& p) const  {
	p << "<any> : ";
    p.dump(type());
}

void Bottom::vdump(Printer& p) const  {
	p << "<bottom> : ";
    p.dump(type());
}

void PrimLit::vdump(Printer& p) const  {
	switch (primtype_kind()) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: p.o << box().get_##T(); break;
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}

    p << " : ";
    p.dump(type());
}

static void dumpNameAndType(Printer& p, const Def* def, const char* name) {
	p << name << " : ";
    p.dump(def->type());
    p << " (";
}

// PrimOp

void BinOp::vdump(Printer& p) const  {
    const char* name;
	switch (node_kind()) {
#define ANYDSL_ARITHOP(op) case Node_##op: name = #op; break;
#include "anydsl/tables/arithoptable.h"
#define ANYDSL_RELOP(op)   case Node_##op: name = #op; break;
#include "anydsl/tables/reloptable.h"
        default: ANYDSL_UNREACHABLE;
	}

    dumpNameAndType(p, this, name);

	p.dump(lhs()) << ", ";
	p.dump(rhs());
	p << ")";
}

void ConvOp::vdump(Printer& p) const {
    const char* name;
    switch (convop_kind()) {
#define ANYDSL_CONVOP(op) case Node_##op: name = #op; break;
#include "anydsl/tables/convoptable.h"
        default: ANYDSL_UNREACHABLE;
    }

    dumpNameAndType(p, this, name);

    p.dump(from());
    p << ')';
}

void Select::vdump(Printer& p) const  {
    dumpNameAndType(p, this, "select");
	p.dump(cond());
	p << ", ";
	p.dump(tval());
	p << ", ";
	p.dump(fval());
	p << ")";
}

void Extract::vdump(Printer& p) const  {
    dumpNameAndType(p, this, "extract");
	p.dump(tuple());
	p << ", " << index() << ")";
}

void Insert::vdump(Printer& p) const  {
    dumpNameAndType(p, this, "insert");
    p << '(';
	p.dump(tuple());
	p << ", " << index() << ", ";
	p.dump(value());
	p << ")";
}

void Tuple::vdump(Printer& p) const {
    p << '(';
    p.dump(type());
    p << ')';
	p << "{";
	ANYDSL_DUMP_COMMA_LIST(p, ops());
	p << "}";
}

// Types

void CompoundType::dumpInner(Printer& p) const  {
	p << "(";
	ANYDSL_DUMP_COMMA_LIST(p, elems());
	p << ")";
}

void PrimType::vdump(Printer& p) const  {
	switch (primtype_kind()) {
#define ANYDSL_UF_TYPE(T) case Node_PrimType_##T: p << #T; return;
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

void Sigma::vdump(Printer& p) const  {
	p << "sigma";
	dumpInner(p);
}

void Pi::vdump(Printer& p) const  {
	p << "pi";
	dumpInner(p);
}

void Lambda::vdump(Printer& p) const  {
	p.dumpName(this);
    p << " : ";
    p.dump(type());
}

void Param::vdump(Printer &p) const  {
	p.dumpName(this);
    p << " : ";
    p.dump(type());
}

//------------------------------------------------------------------------------

void Def::dump() const {
    dump(false);
}

void Def::dump(bool fancy) const {
    Printer p(std::cout, fancy);
    vdump(p);
    std::cout << std::endl;
}

void Lambda::dump(bool fancy, int indent) const  {
    Printer p(std::cout, fancy);

    p.indent += indent;
    p.newline();

	p.dumpName(this);
	p << " = lambda(";
    ANYDSL_DUMP_COMMA_LIST(p, params());
	p << ") : ";
    p.dump(type());
    if (is_extern())
        p << " extern ";
	p.up();

    if (empty())
        p << "jump <EMPTY>";
    else {
        p << "jump(";
        p.dump(to());
        p << ", [";
        ANYDSL_DUMP_COMMA_LIST(p, args());
        p  << "])";
    }
	p.down();

    p.indent -= indent;
}

//------------------------------------------------------------------------------

} // namespace anydsl
