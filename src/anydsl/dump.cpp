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
        for (get_clean_type<BOOST_TYPEOF((list))>::type::const_iterator i = (list).begin(), e = (list).end() - 1; i != e; ++i) { \
            (*i)->dump(printer, mode); \
            printer << ", "; \
        } \
        ((list).back())->dump(printer, mode); \
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
    Printer& operator<<(T& data) {
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

void BinOp::dump(Printer& printer, LambdaPrinterMode mode) const  {
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
	ldef()->dump(printer, mode);
	printer << ", ";
	rdef()->dump(printer, mode);
	printer << ")";
}

// Literal

void Undef::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "<undef>";
}

void ErrorLit::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "<error>";
}

void PrimLit::dump(Printer& printer, LambdaPrinterMode mode) const  {
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

void Goto::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "goto(";
	to()->dump(printer, mode);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, args());
	printer  << "])";
}

void Branch::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "branch(";
    cond()->dump(printer, mode);
	printer << ", ";

	tto()->dump(printer, mode);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, targs());
	printer  << "]), ";

	fto()->dump(printer, mode);
	printer << ", [";
	ANYDSL_DUMP_COMMA_LIST(printer, fargs());
	printer  << "])";
}

// PrimOp

void Select::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "select(";
	cond()->dump(printer, mode);
	printer << ", ";
	tdef()->dump(printer, mode);
	printer << ", ";
	fdef()->dump(printer, mode);
	printer << ")";
}

void Extract::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "extract(";
	tuple()->dump(printer, mode);
	printer << ", ";
	elem()->dump(printer, mode);
	printer << ")";
}

void Insert::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "insert(";
	tuple()->dump(printer, mode);
	printer << ", ";
	elem()->dump(printer, mode);
	printer << ", ";
	value()->dump(printer, mode);
	printer << ")";
}

void Tuple::dump(Printer& printer, LambdaPrinterMode mode) const {
	printer << "{";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << "}";
}

// Types

void NoRet::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "noret";
}

void CompoundType::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "(";
	ANYDSL_DUMP_COMMA_LIST(printer, ops());
	printer << ")";
}

void PrimType::dump(Printer& printer, LambdaPrinterMode mode) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: printer << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
	default:
		ANYDSL_UNREACHABLE;
		break;
	}
}

void Sigma::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "sigma";
	CompoundType::dump(printer, mode);
}

void Pi::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer << "pi";
	CompoundType::dump(printer, mode);
}

void Lambda::dump(Printer& printer, LambdaPrinterMode mode) const  {
	printer.dumpName(this);
	if(mode == LAMBDA_PRINTER_MODE_SKIPBODY)
		return;
	printer << " = lambda(";

    if (!params().empty()) {
        Params::const_iterator i = params().begin();
        while (true) {
            Params::const_iterator j = i;
            ++j;

            if (j != params().end()) {
                (*i).def()->dump(printer, mode);
                printer << ", ";
                i = j;
            } else
                break;
        } 
        (*i).def()->dump(printer, mode);
    }

	printer << ")";
	printer.up();
	jump()->dump(printer, mode);
	printer.down();
}

void Param::dump(Printer &printer, LambdaPrinterMode mode) const  {
	printer.dumpName(this);
}

//------------------------------------------------------------------------------

void AIRNode::dump() const {
    Printer p(std::cout);
    dump(p, LAMBDA_PRINTER_MODE_DEFAULT);
}

//------------------------------------------------------------------------------

} // namespace anydsl
