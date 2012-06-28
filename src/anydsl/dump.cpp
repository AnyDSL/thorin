#include "anydsl/dump.h"

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
        get_clean_type<BOOST_TYPEOF((list))>::type::const_iterator i = (list).begin(); \
        while (true) { \
            get_clean_type<BOOST_TYPEOF((list))>::type::const_iterator j = i; \
            ++j; \
            if (j != (list).end()) { \
                (*i)->dump(printer, descent); \
                printer << ", "; \
                i = j; \
            } else \
                break; \
        }  \
        (*i)->dump(printer, descent); \
    }

namespace anydsl {

class Printer {
public:

    Printer(std::ostream& o)
        : o(o)
        , fancy_(false)
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
        o << "\e[" << code << "m";
    }

    o << n;
    if (!n->debug.empty())
        o << "_[" << n->debug << ']';

    if (fancy())
        o << "\e[m";
}

void BinOp::dump(Printer& printer, bool descent) const  {
	switch(indexKind()) {
#define ANYDSL_ARITHOP(op) case Index_##op: printer << #op;
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: printer << #op;
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) case Index_##op:
#include "anydsl/tables/convoptable.h"
        ANYDSL_NOT_IMPLEMENTED;

	default:
		ANYDSL_UNREACHABLE;
		break;
	}
	printer << "(";
	ldef()->dump(printer, descent);
	printer << ", ";
	rdef()->dump(printer, descent);
	printer << ")";
}

// Literal

void Undef::dump(Printer& printer, bool descent) const  {
	printer << "<undef>";
}

void ErrorLit::dump(Printer& printer, bool descent) const  {
	printer << "<error>";
}

void PrimLit::dump(Printer& printer, bool descent) const  {
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

void Goto::dump(Printer& printer, bool descent) const  {
	printer << "goto(";
	to()->dump(printer, descent);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, args());
	printer  << "])";
}

void Branch::dump(Printer& printer, bool descent) const  {
	printer << "branch(";
    cond()->dump(printer, descent);
	printer << ", ";

	tto()->dump(printer, descent);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, targs());
	printer  << "]), ";

	fto()->dump(printer, descent);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, fargs());
	printer  << "])";
}

// PrimOp

void Select::dump(Printer& printer, bool descent) const  {
	printer << "select(";
	cond()->dump(printer, descent);
	printer << ", ";
	tdef()->dump(printer, descent);
	printer << ", ";
	fdef()->dump(printer, descent);
	printer << ")";
}

void Extract::dump(Printer& printer, bool descent) const  {
	printer << "extract(";
	tuple()->dump(printer, descent);
	printer << ", ";
	elem()->dump(printer, descent);
	printer << ")";
}

void Insert::dump(Printer& printer, bool descent) const  {
	printer << "insert(";
	tuple()->dump(printer, descent);
	printer << ", ";
	elem()->dump(printer, descent);
	printer << ", ";
	value()->dump(printer, descent);
	printer << ")";
}

void Tuple::dump(Printer& printer, bool descent) const {
	printer << "{";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << "}";
}

// Types

void NoRet::dump(Printer& printer, bool descent) const  {
	printer << "noret";
}

void CompoundType::dump(Printer& printer, bool descent) const  {
	printer << "(";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << ")";
}

void PrimType::dump(Printer& printer, bool descent) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: printer << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

void Sigma::dump(Printer& printer, bool descent) const  {
	printer << "sigma";
	CompoundType::dump(printer, descent);
}

void Pi::dump(Printer& printer, bool descent) const  {
	printer << "pi";
	CompoundType::dump(printer, descent);
}

void Lambda::dump(Printer& printer, bool descent) const  {
	printer.dumpName(this);
	if (!descent)
		return;
	printer << " = lambda(";
    ANYDSL_DUMP_COMMA_LIST(printer, params());
	printer << ")";
	printer.up();
	jump()->dump(printer, descent);
	printer.down();
}

void Param::dump(Printer &printer, bool descent) const  {
	printer.dumpName(this);
}

//------------------------------------------------------------------------------

void AIRNode::dump() const {
    Printer p(std::cout);
    dump(p, false);
}

//------------------------------------------------------------------------------

} // namespace anydsl
