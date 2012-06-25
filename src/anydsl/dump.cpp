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

#define ANYDSL_DUMP_COMMA_LIST(list) \
    if (!(list).empty()) { \
        for (get_clean_type<BOOST_TYPEOF((list))>::type::const_iterator i = (list).begin(), e = (list).end() - 1; i != e; ++i) { \
            dump(*i); \
            o << ", "; \
        } \
        dump((list).back()); \
    }

namespace anydsl {

class Printer {
public:

    Printer(std::ostream& o)
        : o(o)
    {}

    void dump(const AIRNode* n);
    void dump(const CompoundType* ct, const char* str);
    void dumpBinOp(const std::string& str, const AIRNode* n);
    void dumpName(const AIRNode* n);

    void newline();
    void up();
    void down();

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
    o << n;
    if (!n->debug.empty())
        o << '[' << n->debug << ']';
}

void Printer::dumpBinOp(const std::string& str, const AIRNode* n) {
    const BinOp* b = n->as<BinOp>();
    o << str << "("; 
    dump(b->ldef());
    o << ", ";
    dump(b->rdef());
    o << ")";
    return;
}

void Printer::dump(const CompoundType* ct, const char* str) {
    o << str << '(';
    ANYDSL_DUMP_COMMA_LIST(ct->ops());
    o << ')';
    return;
}

void Printer::dump(const AIRNode* n) {
    std::string str;

    switch (n->index()) {
/*
 * types
 */

#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: o << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Sigma: 
            return dump(n->as<CompoundType>(), "sigma");

        case Index_Pi: 
            return dump(n->as<CompoundType>(), "pi");

/*
 * literals
 */

#define ANYDSL_U_TYPE(T) case Index_PrimLit_##T: o << n->as<PrimLit>()->box().get_##T(); return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Undef:    o << "<undef>"; return;
        case Index_ErrorLit: o << "<error>"; return;

/*
 * primops
 */

#define ANYDSL_ARITHOP(op) case Index_##op: return dumpBinOp(#op, n);
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: return dumpBinOp(#op, n);
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) case Index_##op:
#include "anydsl/tables/convoptable.h"
        ANYDSL_NOT_IMPLEMENTED;

        case Index_Proj:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_Insert:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_Select:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_Jump: {
            const Jump* jump = n->as<Jump>();
            o << jump->to() << '(';
            ANYDSL_DUMP_COMMA_LIST(jump->args());
            o << ')';
            return;
        }

        case Index_Tuple:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_NoRet: {
            const NoRet* noret = n->as<NoRet>();
            dump(noret->pi());
            return;
        }

/*
 * Param
 */
        case Index_Param:
            dumpName(n);
            return;

/*
 * Lambda
 */
        case Index_Lambda: {
            const Lambda* lambda = n->as<Lambda>();
            const Params& params = lambda->params();

            dumpName(lambda);
            o << " = lambda(";
            ANYDSL_DUMP_COMMA_LIST(params);
            o << ')';
            up();
            dump(lambda->jump());
            down();
            return;
        }

        //default: ANYDSL_NOT_IMPLEMENTED;
    }
}

//------------------------------------------------------------------------------

void dump(const AIRNode* n, std::ostream& o /*= std::cout*/) {
    Printer p(o);
    p.dump(n);
}

//------------------------------------------------------------------------------

} // namespace anydsl
