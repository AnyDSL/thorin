//#include "anydsl/dump.h"

#include "anydsl/airnode.h"
#include "anydsl/literal.h"
#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

class Dumper {
public:
    void dump(const AIRNode* n, std::ostream& s);
};

void dump(const AIRNode* n, std::ostream& s /*= std::cout*/) {
    Dumper p;
    p.dump(n, s);
}

static void dumpCompoundType(const std::string& str, const AIRNode* n, std::ostream& s) {
    const Sigma* sigma = n->as<Sigma>();
    s << str << '(';

    if (!sigma->types().empty()) {
        for (Sigma::Types::const_iterator i = sigma->types().begin(), 
                                          e = sigma->types().end() - 1; 
                                          i != e; ++i) {
            dump(*i, s);
            s << ", ";
        }

        dump(sigma->types().back(), s);
    }

    s << ')';

    return;
}

static void dumpBinOp(const std::string& str, const AIRNode* n, std::ostream& s) {
    const BinOp* b = n->as<BinOp>();
    s << str << "("; 
    dump(b->luse().def(), s);
    s << ", ";
    dump(b->ruse().def(), s);
    s << ")";
    return;
}

void Dumper::dump(const AIRNode* n, std::ostream& s) {
    std::string str;

    switch (n->index()) {
/*
 * Use
 */
        // normally we follow a use to a def
        case Index_Use: 
            return dump(n->as<Use>()->def(), s);

/*
 * types
 */

#define ANYDSL_U_TYPE(T) case Index_PrimType_##T: s << #T; return;
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"

        case Index_Sigma: return dumpCompoundType("sigma", n, s);
        case Index_Pi:    return dumpCompoundType("pi",    n->as<Pi>()->sigma(), s);

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

#define ANYDSL_ARITHOP(op) case Index_##op: return dumpBinOp(#op, n, s);
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op)   case Index_##op: return dumpBinOp(#op, n, s);
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
            dump(noret->pi(), s);
            return;
        }

/*
 * Param
 */
        case Index_Params:
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
