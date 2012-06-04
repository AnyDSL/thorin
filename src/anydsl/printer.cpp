#include "anydsl/printer.h"

#include "anydsl/airnode.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

class Printer {
public:
    void print(std::ostream& s, const AIRNode* n);
};

void print(std::ostream& s, const AIRNode* n) {
    Printer p;
    p.print(s, n);
}

static void printCompoundType(std::ostream& s, const std::string& str, const AIRNode* n) {
    const CompoundType* c = n->as<CompoundType>();
    s << str << '(';

    if (!c->types().empty()) {
        for (Types::const_iterator i = c->types().begin(), e = --c->types().end(); i != e; ++i) {
            print(s, *i);
            s << ", ";
        }

        print(s, c->types().back());
    }

    s << ')';

    return;
}

static void printBinOp(std::ostream& s, const std::string& str, const AIRNode* n) {
    const BinOp* b = n->as<BinOp>();
    s << str << "("; 
    print(s, b->luse().def());
    s << ", ";
    print(s, b->ruse().def());
    s << ")";
    return;
}

void Printer::print(std::ostream& s, const AIRNode* n) {
    std::string str;

    switch (n->index()) {
/*
 * Use
 */
        // normally we follow a use to a def
        case Index_Use: 
            return print(s, n->as<Use>()->def());

/*
 * types
 */

#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: s << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Sigma: return printCompoundType(s, "sigma", n);
        case Index_Pi:    return printCompoundType(s, "pi",    n);

/*
 * literals
 */

#define ANYDSL_U_TYPE(T) case Index_PrimLit_##T: s << n->as<PrimLit>()->box().get_##T(); return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Undef:    s << "<undef>"; return;
        case Index_ErrorLit: s << "<error>"; return;

/*
 * primops
 */

#define ANYDSL_ARITHOP(op) case Index_##op: return printBinOp(s, #op, n);
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: return printBinOp(s, #op, n);
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

        case Index_Jump:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_Tuple:
            ANYDSL_NOT_IMPLEMENTED;

        case Index_NoRet:
        {
            const NoRet* noret = n->as<NoRet>();
            print(s, noret->pi());
            return;
        }

/*
 * Param
 */
        case Index_Param:
            ANYDSL_NOT_IMPLEMENTED;

/*
 * Lambda
 */
        case Index_Lambda:
            ANYDSL_NOT_IMPLEMENTED;

        //default: ANYDSL_NOT_IMPLEMENTED;
    }
}

} // namespace anydsl
