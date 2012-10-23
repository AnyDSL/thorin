#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <iostream>

#include "anydsl2/cfg.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/ref.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

#include "impala/type.h"

using anydsl2::Array;
using anydsl2::ArrayRef;
using anydsl2::BB;
using anydsl2::Def;
using anydsl2::RVal;
using anydsl2::Ref;
using anydsl2::Symbol;
using anydsl2::TupleRef;
using anydsl2::Var;
using anydsl2::VarRef;
using anydsl2::World;
using anydsl2::make_name;
using anydsl2::u32;

namespace impala {

typedef boost::unordered_map<Symbol, anydsl2::Fct*> FctMap;

class CodeGen {
public:

    CodeGen(World& world)
        : world(world)
    {}

    void fixto(BB* to) {
        if (reachable())
            curBB->fixto(to);
    }

    bool reachable() const { return curBB; }

    BB* curBB;
    anydsl2::Fct* curFct;
    World& world;
    FctMap fcts;
};

//------------------------------------------------------------------------------

void emit(World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
    //cg.world.cleanup();
}

//------------------------------------------------------------------------------

void Prg::emit(CodeGen& cg) const {
    for_all (f, fcts()) {
        //for_all (generic, f->generics()) {
        //}
        size_t size = f->lambda().params().size();
        Array<const anydsl2::Type*> tparams(size);
        Array<Symbol>      sparams(size);

        size_t i = 0;
        for_all (param, f->lambda().params()) {
            tparams[i] = param->type()->convert(cg);
            sparams[i] = param->symbol();
            ++i;
        }

        const anydsl2::Type* ret = f->lambda().pi()->ret()->convert(cg);
        cg.fcts[f->symbol()] = new anydsl2::Fct(cg.world, tparams, sparams, ret, f->symbol().str());
    }

    for_all (f, fcts()) {
        cg.curBB = cg.curFct = cg.fcts[f->symbol()];
        f->emit(cg);
    }

    for_all (p, cg.fcts)
        delete p.second;
}

void Lambda::emit(CodeGen& cg, const char* what) const {
    body()->emit(cg);

    if (continuation()) {
        if (cg.reachable())
            std::cerr << what << " does not end with a call\n";
    } else
        cg.fixto(cg.curFct->exit());

    cg.curFct->emit();

    cg.curFct = 0;
    cg.curBB = 0;
}

void Fct::emit(CodeGen& cg) const {
    lambda().emit(cg, symbol().str());
}

Var* Decl::emit(CodeGen& cg) const {
    Var* var = cg.curBB->insert(symbol(), cg.world.bottom(type()->convert(cg)));
    var->load()->debug = symbol().str();

    return var;
}

/*
 * Expr
 */

Array<const Def*> Expr::emit_ops(CodeGen& cg) const {
    Array<const Def*> ops(ops_.size());

    size_t i = 0;
    for_all (op, ops_)
        ops[i++] = op->emit(cg)->load();

    return ops;
}

RefPtr EmptyExpr::emit(CodeGen& cg) const { 
    return Ref::create(cg.world.bottom(cg.world.unit())); 
}

RefPtr Literal::emit(CodeGen& cg) const {
    anydsl2::PrimTypeKind akind;

    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: akind = anydsl2::PrimType_##atype; break;
#include "impala/tokenlist.h"
        case LIT_bool: akind = anydsl2::PrimType_u1; break;
        default: ANYDSL2_UNREACHABLE;
    }

    return Ref::create(cg.world.literal(akind, box()));
}

RefPtr LambdaExpr::emit(CodeGen& cg) const {
    lambda().emit(cg, "anonymous lambda expression");
    return Ref::create((const Def*) 0);
}

RefPtr Tuple::emit(CodeGen& cg) const {
    Array<const Def*> vals = emit_ops(cg);

    return Ref::create(cg.world.tuple(vals));
}

RefPtr Id::emit(CodeGen& cg) const {
    if (type()->isa<Pi>()) {
        FctMap::iterator i = cg.fcts.find(symbol());
        if (i != cg.fcts.end())
            return Ref::create(i->second->top());
    }

    return Ref::create(cg.curBB->lookup(symbol(), type()->convert(cg)));
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const anydsl2::PrimType* pt = def->type()->as<anydsl2::PrimType>();
            const anydsl2::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
            const Def* ndef = cg.world.arithop(Token::toArithOp((TokenKind) kind()), def, one);
            ref->store(ndef);

            return ref;
        }
        case ADD:
            return rhs()->emit(cg); // this is a NOP

        case SUB: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const anydsl2::PrimType* pt = def->type()->as<anydsl2::PrimType>();
            const anydsl2::PrimLit* zero; 

            switch (pt->primtype_kind()) {
                case anydsl2::PrimType_f32: zero = cg.world.literal_f32(-0.f); break;
                case anydsl2::PrimType_f64: zero = cg.world.literal_f64(-0.0); break;
                default: 
                    assert(pt->is_int()); 
                    zero = cg.world.literal(pt->primtype_kind(), 0u);
            }

            return Ref::create(cg.world.arithop(anydsl2::ArithOp_sub, zero, def));
        }
        default: ANYDSL2_UNREACHABLE;
    }
}

RefPtr InfixExpr::emit(CodeGen& cg) const {
    TokenKind op = (TokenKind) kind();

    if (Token::is_asgn(op)) {
        const Id* id = lhs()->isa<Id>();

        // special case for 'a = expr' -> don't use lookup!
        RefPtr lref = op == Token::ASGN && id
                ? Ref::create(cg.curBB->insert(id->symbol(), cg.world.bottom(id->type()->convert(cg))))
                : lhs()->emit(cg);

        const Def* ldef = lref->load();
        const Def* rdef = rhs()->emit(cg)->load();

        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            rdef = cg.world.binop(Token::toBinOp(sop), ldef, rdef);
        }

        lref->store(rdef);

        return lref;
    }

    const Def* ldef = lhs()->emit(cg)->load();
    const Def* rdef = rhs()->emit(cg)->load();

    return Ref::create(cg.world.binop(Token::toBinOp(op), ldef, rdef));
}

RefPtr PostfixExpr::emit(CodeGen& cg) const {
    RefPtr ref = lhs()->emit(cg);
    const Def* def = ref->load();
    const anydsl2::PrimType* pt = def->type()->as<anydsl2::PrimType>();
    const anydsl2::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
    const Def* ndef = cg.world.arithop(Token::toArithOp((TokenKind) kind()), def, one);
    ref->store(ndef);

    return Ref::create(def);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    u32 pos = index()->as<Literal>()->box().get_u32();

    return Ref::create(lhs()->emit(cg), pos);
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);
    const anydsl2::Type* ret = type()->convert(cg);

    if (ret)
        return Ref::create(cg.curBB->call(ops[0], ops.slice_back(1), ret));
    else {
        cg.curBB->tail_call(ops[0], ops.slice_back(1));
        cg.curBB = 0;
        return RefPtr(0);
    }
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    Var* var = decl()->emit(cg);

    if (const Expr* init_expr = init())
        var->store(init_expr->emit(cg)->load());
}

void ExprStmt::emit(CodeGen& cg) const {
    if (cg.reachable())
        expr()->emit(cg);
}

void IfElseStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int curId = id++;

    // always create elseBB -- the edge from headBB to nextBB would be crtical anyway

    // create BBs
    BB* thenBB = cg.curFct->createBB(make_name("if-then", curId));
    BB* elseBB = cg.curFct->createBB(make_name("if-else", curId));

    // condition
    if (cg.reachable()) {
        const Def* c = cond()->emit(cg)->load();
        cg.curBB->branch(c, thenBB, elseBB);
    }

    thenBB->seal();
    elseBB->seal();

    // then
    cg.curBB = thenBB;
    thenStmt()->emit(cg);
    BB* thenCur = cg.curBB;

    // else
    cg.curBB = elseBB;
    elseStmt()->emit(cg);
    BB* elseCur = cg.curBB;

    if (!elseCur) {
        cg.curBB = thenCur;
    } else if (thenCur) {
        BB* nextBB = cg.curFct->createBB(make_name("if-next", curId));
        thenCur->fixto(nextBB);
        elseCur->fixto(nextBB);
        nextBB->seal();
        cg.curBB = nextBB;
    }
}

void WhileStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int curId = id++;

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("while-head", curId));
    BB* bodyBB = cg.curFct->createBB(make_name("while-body", curId));
    BB* nextBB = cg.curFct->createBB(make_name("while-next", curId));

    // head
    cg.fixto(headBB);
    cg.curBB = headBB;

    // condition
    const Def* c = cond()->emit(cg)->load();
    headBB->branch(c, bodyBB, nextBB);
    bodyBB->seal();

    // body
    cg.curBB = bodyBB;
    body()->emit(cg);
    cg.fixto(headBB);
    headBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();
}

void DoWhileStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int curId = id++;

    // create BBs
    BB* bodyBB = cg.curFct->createBB(make_name("dowhile-body", curId));
    BB* condBB = cg.curFct->createBB(make_name("dowhile-cond", curId));
    BB* critBB = cg.curFct->createBB(make_name("dowhile-crit", curId));
    BB* nextBB = cg.curFct->createBB(make_name("dowhile-next", curId));

    // body
    cg.fixto(bodyBB);
    cg.curBB = bodyBB;
    body()->emit(cg);

    // condition
    cg.fixto(condBB);
    condBB->seal();
    cg.curBB = condBB;
    const Def* c = cond()->emit(cg)->load();
    condBB->branch(c, critBB, nextBB);
    critBB->seal();
    cg.curBB = critBB;
    critBB->jump(bodyBB);
    bodyBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();
}

void ForStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int curId = id++;

    init()->emit(cg);

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("for-head", curId));
    BB* bodyBB = cg.curFct->createBB(make_name("for-body", curId));
    BB* stepBB = cg.curFct->createBB(make_name("for-step", curId));
    BB* nextBB = cg.curFct->createBB(make_name("for-next", curId));

    // head
    cg.fixto(headBB);
    cg.curBB = headBB;

    // condition
    const Def* c = cond()->emit(cg)->load();
    headBB->branch(c, bodyBB, nextBB);
    bodyBB->seal();

    // body
    cg.curBB = bodyBB;
    body()->emit(cg);
    cg.fixto(stepBB);
    stepBB->seal();

    // step
    cg.curBB = stepBB;
    step()->emit(cg);
    cg.fixto(headBB);
    headBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();
}

void BreakStmt::emit(CodeGen& cg) const {
}

void ContinueStmt::emit(CodeGen& cg) const {
}

void ReturnStmt::emit(CodeGen& cg) const {
    if (const Call* call = expr()->isa<Call>()) {
        Array<const Def*> ops = call->emit_ops(cg);
        cg.curBB->return_call(ops[0], ops.slice_back(1));
    } else {
        const Def* def = expr()->emit(cg)->load();
        cg.curBB->insert(Symbol("<result>"), def);
        cg.curBB->jump(cg.curFct->exit());
    }

    // all statements in the same BB are unreachable
    cg.curBB = 0;
}

void ScopeStmt::emit(CodeGen& cg) const {
    for_all (const &s, stmts())
        s->emit(cg);
}

//------------------------------------------------------------------------------

/*
 * Type
 */

const anydsl2::Type* PrimType::convert(CodeGen& cg) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) \
        case TYPE_##itype: \
            return cg.world.type_##atype();
#include "impala/tokenlist.h"
        default:
            ANYDSL2_UNREACHABLE;
    }
}

const anydsl2::Type* Void::convert(CodeGen& cg) const { return cg.world.unit(); }
const anydsl2::Type* NoRet::convert(CodeGen&) const { return 0; }
const anydsl2::Type* TypeError::convert(CodeGen&) const { ANYDSL2_UNREACHABLE; }

const anydsl2::Type* Pi::convert(CodeGen& cg) const {
    size_t size = elems().size();
    Array<const anydsl2::Type*> types(size + 1);

    for (size_t i = 0; i < size; ++i)
        types[i] = elems()[i]->convert(cg);

    if (!ret()->isa<NoRet>())
        types[size++] = cg.world.pi1(ret()->convert(cg));

    return cg.world.pi(types.slice_front(size));
}

const anydsl2::Type* Sigma::convert(CodeGen& cg) const {
    size_t size = elems().size();
    Array<const anydsl2::Type*> types(size);

    for (size_t i = 0; i < size; ++i)
        types[i] = elems()[i]->convert(cg);

    return cg.world.sigma(types);
}

const anydsl2::Type* Generic::convert(CodeGen& cg) const {
    // TODO
    return 0;
    //return cg.fcts[fct()->symbol()]->top_->append_generic();
}

} // namespace impala
