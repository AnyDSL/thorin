#include <boost/typeof/typeof.hpp>

#include "anydsl/airnode.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/for_all.h"

template<class T>
struct get_clean_type { typedef T type; } ;

template<class T>
struct get_clean_type<const T&> {typedef T type; };

#define ANYDSL_DUMP_COMMA_LIST(printer, list) \
    if (!(list).empty()) { \
        for (get_clean_type<BOOST_TYPEOF((list))>::type::const_iterator i = (list).begin(), e = (list).end() - 1; i != e; ++i) { \
            (*i)->vdump(printer); \
            printer << ", "; \
        } \
        ((list).back())->vdump(printer); \
    }

namespace anydsl {

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , fancy_(fancy)
        , indent_(0)
    {}

    bool fancy() const { return fancy_; }

    void dump(const AIRNode* n);
    void dumpName(const AIRNode* n);

    void newline();
    void up();
    void down();

    template<class T>
    Printer& operator << (const T& data) {
    	o << data;
    	return *this;
    }

    std::ostream& o;

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

void Printer::dumpName(const AIRNode* n) {
    if (fancy()) {
        unsigned i = uintptr_t(n);
        unsigned sum = 0;

        while (i) {
            sum += i & 0x3;
            i >>= 2;
        }

        sum += i;

        // elide white = 0 and black = 7
        int code = (sum % 6) + 30 + 1;
        o << "\33[" << code << "m";
    }

    o << n;
    if (!n->debug.empty())
        o << "_[" << n->debug << ']';

    if (fancy())
        o << "\33[m";
}

void BinOp::vdump(Printer& printer) const  {
	switch(indexKind()) {
#define ANYDSL_ARITHOP(op) case Index_##op: printer << #op; break;
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: printer << #op; break;
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) case Index_##op: ANYDSL_NOT_IMPLEMENTED; break;
#include "anydsl/tables/convoptable.h"
        
	default:
		ANYDSL_UNREACHABLE;
	}

	printer << "(";
	ldef()->vdump(printer);
	printer << ", ";
	rdef()->vdump(printer);
	printer << ")";
}

// Literal

void Undef::vdump(Printer& printer) const  {
	printer << "<undef>";
}

void ErrorLit::vdump(Printer& printer) const  {
	printer << "<error>";
}

void PrimLit::vdump(Printer& printer) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimLit_##T: printer.o << box().get_##T(); return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

// Jump

void Goto::vdump(Printer& printer) const  {
	printer << "goto(";
	to()->vdump(printer);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, args());
	printer  << "])";
}

void Branch::vdump(Printer& printer) const  {
	printer << "branch(";
    cond()->vdump(printer);
	printer << ", ";

	tto()->vdump(printer);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, targs());
	printer  << "]), ";

	fto()->vdump(printer);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, fargs());
	printer  << "])";
}

// PrimOp

void Select::vdump(Printer& printer) const  {
	printer << "select(";
	cond()->vdump(printer);
	printer << ", ";
	tdef()->vdump(printer);
	printer << ", ";
	fdef()->vdump(printer);
	printer << ")";
}

void Extract::vdump(Printer& printer) const  {
	printer << "extract(";
	tuple()->vdump(printer);
	printer << ", " << index() << ")";
}

void Insert::vdump(Printer& printer) const  {
	printer << "insert(";
	tuple()->vdump(printer);
	printer << ", " << index() << ", ";
	value()->vdump(printer);
	printer << ")";
}

void Tuple::vdump(Printer& printer) const {
	printer << "{";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << "}";
}

// Types

void NoRet::vdump(Printer& printer) const  {
	printer << "noret";
}

void CompoundType::dump(Printer& printer) const  {
	printer << "(";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << ")";
}

void PrimType::vdump(Printer& printer) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: printer << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

void Sigma::vdump(Printer& printer) const  {
	printer << "sigma";
	CompoundType::dump(printer);
}

void Pi::vdump(Printer& printer) const  {
	printer << "pi";
	CompoundType::dump(printer);
}

void Lambda::vdump(Printer& printer) const  {
	printer.dumpName(this);
}

void Param::vdump(Printer &printer) const  {
	printer.dumpName(this);
}

//------------------------------------------------------------------------------

void AIRNode::dump(bool fancy) const {
    Printer printer(std::cout, fancy);
    vdump(printer);
}

void Lambda::dump(bool fancy) const  {
    Printer printer(std::cout, fancy);

	printer.dumpName(this);
	printer << " = lambda(";
    ANYDSL_DUMP_COMMA_LIST(printer, params());
	printer << ")";
	printer.up();
	jump()->vdump(printer);
	printer.down();
}

//------------------------------------------------------------------------------

} // namespace anydsl
