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

void print(std::ostream& s, AIRNode* n) {
    Printer p;
    p.print(s, n);
}

void Printer::print(std::ostream& s, const AIRNode* n) {
    std::string str;

    switch (n->index()) {
        // normally we follow a use to a def
        case Index_Use: 
            return print(s, n->as<Use>()->def());

/*
 * types
 */

#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: s << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Sigma: str = "sigma";
        case Index_Pi:    str = "pi";
        {
            const CompoundType* c = n->as<CompoundType>();
            FOREACH(const& t, c->types())
                print(s, t);
        }

/*
 * literals
 */

#define ANYDSL_U_TYPE(T) case Index_PrimLit_##T: s << n->as<PrimLit>()->box().get_##T(); return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Undef: s << "<undef>"; return;
        case Index_ErrorLit: s << "<error>"; return;

/*
 * primops
 */

#define ANYDSL_ARITHOP(op) case Index_##op: str = #op;
#define ANYDSL_RELOP(op)   case Index_##op: str = #op;
#include "anydsl/tables/arithoptable.h"
#include "anydsl/tables/reloptable.h"
        {
            const BinOp* b = n->as<BinOp>();
            s << str << "("; 
            print(s, b->luse.def());
            s << ", ";
            print(s, b->ruse.def());
            s << ")";
            return;
        }

#define ANYDSL_CONVOP(op) case Index_##op:
#include "anydsl/tables/convoptable.h"
        ANYDSL_NOT_IMPLEMENTED;

        case Index_Proj:
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
