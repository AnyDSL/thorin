#include <boost/typeof/typeof.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/printer.h"

namespace anydsl2 {

// Literal

void Any::vdump(Printer& p) const {
	p << "<any> : ";
    p.dump(type());
}

void Bottom::vdump(Printer& p) const {
	p << "<bottom> : ";
    p.dump(type());
}

void TypeKeeper::vdump(Printer& p) const {
	p << "<typekeeper> : ";
    p.dump(type());
}

void PrimLit::vdump(Printer& p) const {
	switch (primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: p.o << box().get_##T(); break;
#include "anydsl2/tables/primtypetable.h"
	default:
		ANYDSL2_UNREACHABLE;
		break;
	}

    p << " : ";
    p.dump(type());
}

static void dump_name_and_type(Printer& p, const Def* def, const char* name) {
	p << name << " : ";
    p.dump(def->type());
    p << " (";
}

// PrimOp

void BinOp::vdump(Printer& p) const {
    const char* name;
	switch (node_kind()) {
#define ANYDSL2_ARITHOP(op) case Node_##op: name = #op; break;
#include "anydsl2/tables/arithoptable.h"
#define ANYDSL2_RELOP(op)   case Node_##op: name = #op; break;
#include "anydsl2/tables/reloptable.h"
        default: ANYDSL2_UNREACHABLE;
	}

    dump_name_and_type(p, this, name);

	p.dump(lhs()) << ", ";
	p.dump(rhs());
	p << ")";
}

void ConvOp::vdump(Printer& p) const {
    const char* name;
    switch (convop_kind()) {
#define ANYDSL2_CONVOP(op) case Node_##op: name = #op; break;
#include "anydsl2/tables/convoptable.h"
        default: ANYDSL2_UNREACHABLE;
    }

    dump_name_and_type(p, this, name);

    p.dump(from());
    p << ')';
}

void Select::vdump(Printer& p) const {
    dump_name_and_type(p, this, "select");
	p.dump(cond());
	p << ", ";
	p.dump(tval());
	p << ", ";
	p.dump(fval());
	p << ")";
}

void Extract::vdump(Printer& p) const {
    dump_name_and_type(p, this, "extract");
	p.dump(tuple());
	p << ", "; 
    p.dump(index());
    p << ")";
}

void Insert::vdump(Printer& p) const {
    dump_name_and_type(p, this, "insert");
	p.dump(tuple());
	p << ", " ;
    p.dump(index());
    p << ", ";
	p.dump(value());
	p << ")";
}

void Tuple::vdump(Printer& p) const {
    p << '(';
    p.dump(type());
    p << ')';
	p << "{";
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
	p << "}";
}

void Load::vdump(Printer& p) const {
    dump_name_and_type(p, this, "load");
    p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ')';
}

void Store::vdump(Printer& p) const {
    dump_name_and_type(p, this, "store");
    p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ')';
}

void Enter::vdump(Printer& p) const {
    dump_name_and_type(p, this, "enter");
    p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ')';
}

void Leave::vdump(Printer& p) const {
    dump_name_and_type(p, this, "leave");
    p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ')';
}

void Slot::vdump(Printer& p) const {
    dump_name_and_type(p, this, "slot");
    p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, ops());
    p << ')';
}

/*
 * Types
 */

void CompoundType::dump_inner(Printer& p) const {
    if (num_generics()) {
        p << '<';
        ANYDSL2_DUMP_COMMA_LIST(p, generics());
        p << '>';
    }

	p << '(';
	ANYDSL2_DUMP_COMMA_LIST(p, elems());
	p << ')';
}

void Mem::vdump(Printer& p) const { 
    p << "mem"; 
}

void Frame::vdump(Printer& p) const { 
    p << "frame"; 
}

void Ptr::vdump(Printer& p) const {
    ref()->dump();
    p << '*';
}

void PrimType::vdump(Printer& p) const {
	switch (primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case Node_PrimType_##T: p << #T; return;
#include "anydsl2/tables/primtypetable.h"
	default:
		ANYDSL2_UNREACHABLE;
		break;
	}
}

void Sigma::vdump(Printer& p) const {
    // TODO cycles
	p << "sigma";
	dump_inner(p);
}

void Pi::vdump(Printer& p) const {
	p << "pi";
	dump_inner(p);
}

void Generic::vdump(Printer &p) const {
    if (debug.empty())
        p << this;
    else
        p << debug;
}

void Lambda::vdump(Printer& p) const {
	p.dump_name(this);
    p << " : ";
    p.dump(type());
}

void Param::vdump(Printer &p) const {
	p.dump_name(this);
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

void Type::dump() const {
    dump(false);
}

void Type::dump(bool fancy) const {
    Printer p(std::cout, fancy);
    vdump(p);
    std::cout << std::endl;
}

void Lambda::dump(bool fancy, int indent) const {
    Printer p(std::cout, fancy);

    p.indent += indent;
    p.newline();

	p.dump_name(this);
	p << " = lambda(";
    ANYDSL2_DUMP_COMMA_LIST(p, params());
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
        ANYDSL2_DUMP_COMMA_LIST(p, args());
        p  << "])";
    }
	p.down();

    p.indent -= indent;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
