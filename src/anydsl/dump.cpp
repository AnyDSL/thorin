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

#define ANYDSL_DUMP_COMMA_LIST(p, list) \
    const BOOST_TYPEOF((list))& l = (list); \
    if (!l.empty()) { \
        for (get_clean_type<BOOST_TYPEOF(l)>::type::const_iterator i = l.begin(), e = l.end() - 1; i != e; ++i) { \
            p.dump(*i); \
            p << ", "; \
        } \
        p.dump(l.back()); \
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

void Printer::dump(const AIRNode* n) {
    if (n)
        return n->vdump(*this);

    o << "<NULL>";
}

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

    if (!n->debug.empty())
        o << n->debug;
    else
        o << n;

    if (fancy())
        o << "\33[m";
}

// Literal

void Undef::vdump(Printer& p) const  {
	p << "<undef> : ";
    p.dump(type());
}

void ErrorLit::vdump(Printer& p) const  {
	p << "<error>";
}

void PrimLit::vdump(Printer& p) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimLit_##T: p.o << box().get_##T(); break;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
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

// Jump

void Jump::vdump(Printer& p) const  {
    dumpNameAndType(p, this, "jump");
    p.dump(to());
	p << ", [";
	ANYDSL_DUMP_COMMA_LIST(p, args());
	p  << "])";
}

// PrimOp

void BinOp::vdump(Printer& p) const  {
    const char* name;
	switch(indexKind()) {
#define ANYDSL_ARITHOP(op) case Index_##op: name = #op; break;
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: name = #op; break;
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) case Index_##op: ANYDSL_NOT_IMPLEMENTED; break;
#include "anydsl/tables/convoptable.h"
        
	default:
		ANYDSL_UNREACHABLE;
	}

    dumpNameAndType(p, this, name);

	p.dump(ldef());
	p << ", ";
	p.dump(rdef());
	p << ")";
}


void Select::vdump(Printer& p) const  {
    dumpNameAndType(p, this, "select");
	p.dump(cond());
	p << ", ";
	p.dump(tdef());
	p << ", ";
	p.dump(fdef());
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

void NoRet::vdump(Printer& p) const  {
	p << "noret";
}

void CompoundType::dumpInner(Printer& p) const  {
	p << "(";
	ANYDSL_DUMP_COMMA_LIST(p, ops());
	p << ")";
}

void PrimType::vdump(Printer& p) const  {
	switch(indexKind()) {
#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: p << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
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

void AIRNode::dump() const {
    dump(false);
}

void AIRNode::dump(bool fancy) const {
    Printer p(std::cout, fancy);
    vdump(p);
}

void Lambda::dump(bool fancy) const  {
    Printer p(std::cout, fancy);

	p.dumpName(this);
	p << " = lambda(";
    ANYDSL_DUMP_COMMA_LIST(p, params());
	p << ") : ";
    p.dump(type());
	p.up();
    p.dump(jump());
	p.down();
}

//------------------------------------------------------------------------------

} // namespace anydsl
