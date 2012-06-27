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
        , fancy_(false)
        , indent_(0)
    {}

    bool fancy() const { return fancy_; }

    void dump(const AIRNode* n, bool goInsideLambda = false);
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

void Printer::dump(const AIRNode* n, bool goInsideLambda /*= false*/) {
    std::string str;

    switch (n->indexKind()) {
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

        case Index_Extract: {
            const Extract* extract = n->as<Extract>();
            o << "extract(";
            dump(extract->tuple());
            o << ", ";
            dump(extract->elem());
            o << ')';
            return;
        }
        case Index_Insert: {
            const Insert* insert = n->as<Insert>();
            o << "insert(";
            dump(insert->tuple());
            o << ", ";
            dump(insert->elem());
            o << ", ";
            dump(insert->value());
            o << ')';
            return;
        }
        case Index_Select: {
            const Select* select = n->as<Select>();
            o << "select(";
            dump(select->cond());
            o << ", ";
            dump(select->tdef());
            o << ", ";
            dump(select->fdef());
            o << ')';
            return;
        }
        case Index_Jump: {
            const Jump* jump = n->as<Jump>();
            o << "jump(";
            dump(jump->to());
            o << ", [";
            ANYDSL_DUMP_COMMA_LIST(jump->args());
            o << "])";
            return;
        }
        case Index_Tuple: {
            const Tuple* tuple = n->as<Tuple>();
            o << '{';
            ANYDSL_DUMP_COMMA_LIST(tuple->ops());
            o << '}';
            return;
        }
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
        case Index_Lambda: 
            if (goInsideLambda) {
                const Lambda* lambda = n->as<Lambda>();
                //const Params& params = lambda->params();

                dumpName(lambda);
                o << " = lambda(";
                //ANYDSL_DUMP_COMMA_LIST(params);
                o << ')';
                up();
                dump(lambda->jump());
                down();
                return;
            } else {
                dumpName(n);
                return;
            }
    }
}

//------------------------------------------------------------------------------

void dump(const AIRNode* n, std::ostream& o /*= std::cout*/) {
    Printer p(o);
    p.dump(n, true);
}

//------------------------------------------------------------------------------

} // namespace anydsl
