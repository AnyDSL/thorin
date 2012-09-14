#include "impala/ast.h"

#include <boost/unordered_map.hpp>

#include "anydsl/cfg.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

#include "impala/type.h"

using anydsl::Array;
using anydsl::ArrayRef;
using anydsl::BB;
using anydsl::Def;
using anydsl::Var;
using anydsl::World;
using anydsl::make_name;
using anydsl::u32;

namespace impala {

typedef boost::unordered_map<anydsl::Symbol, anydsl::Fct*> FctMap;

class CodeGen {
public:

    CodeGen(anydsl::World& world)
        : world(world)
    {}

    void fixto(BB* to) {
        if (reachable())
            curBB->fixto(to);
    }

    bool reachable() const { return curBB; }

    const anydsl::Type* convert(const Type* type) { return type->convert(world); }

    anydsl::BB* curBB;
    anydsl::Fct* curFct;
    anydsl::World& world;
    FctMap fcts;
};

//------------------------------------------------------------------------------

void emit(anydsl::World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
    //cg.world.cleanup();
}

//------------------------------------------------------------------------------

void Prg::emit(CodeGen& cg) const {
    for_all (f, fcts()) {
        size_t size = f->params().size();
        Array<const anydsl::Type*> tparams(size);
        Array<anydsl::Symbol>      sparams(size);

        size_t i = 0;
        for_all (param, f->params()) {
            tparams[i] = cg.convert(param->type());
            sparams[i] = param->symbol();
            ++i;
        }

        const anydsl::Type* ret = cg.convert(f->pi()->ret());
        cg.fcts[f->symbol()] = new anydsl::Fct(cg.world, tparams, sparams, ret, f->symbol().str());
    }

    for_all (f, fcts()) {
        cg.curBB = cg.curFct = cg.fcts[f->symbol()];
        f->emit(cg);
    }

    for_all (p, cg.fcts)
        delete p.second;
}

void Fct::emit(CodeGen& cg) const {

    body()->emit(cg);

    if (continuation()) {
        if (cg.reachable())
            std::cerr << symbol() << " does not end with a call\n";
    } else
        cg.fixto(cg.curFct->exit());

    cg.curFct->emit();

    cg.curFct = 0;
    cg.curBB = 0;
}

Var* Decl::emit(CodeGen& cg) const {
    Var* var = cg.curBB->insert(symbol(), cg.world.bottom(cg.convert(type())));
    var->load()->debug = symbol().str();

    return var;
}

/*
 * Expr
 */

const Def* EmptyExpr::remit(CodeGen& cg) const {
    return cg.world.bottom(cg.world.unit());
}

const Def* Literal::remit(CodeGen& cg) const {
    anydsl::PrimTypeKind akind;

    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: akind = anydsl::PrimType_##atype; break;
#include "impala/tokenlist.h"
        case LIT_bool: akind = anydsl::PrimType_u1; break;
        default: ANYDSL_UNREACHABLE;
    }

    return cg.world.literal(akind, box());
}

const Def* Tuple::remit(CodeGen& cg) const {
    Array<const Def*> vals(ops().size());

    size_t i = 0;
    for_all (op, ops())
        vals[i++] = op->remit(cg);

    return cg.world.tuple(vals);
}

Var* Id::lemit(CodeGen& cg) const {
    return cg.curBB->lookup(symbol(), cg.convert(type()));
}

const Def* Id::remit(CodeGen& cg) const {
    if (type()->isa<Pi>()) {
        FctMap::iterator i = cg.fcts.find(symbol());
        if (i == cg.fcts.end())
            return lemit(cg)->load();

        return i->second->top();
    }

    return lemit(cg)->load();
}

Var* PrefixExpr::lemit(CodeGen& cg) const {
    assert(kind() == INC || kind() == DEC);

    Var* var = rhs()->lemit(cg);
    const Def* def = var->load();
    const anydsl::PrimType* pt = def->type()->as<anydsl::PrimType>();
    const anydsl::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
    const Def* ndef = cg.world.arithop(Token::toArithOp((TokenKind) kind()), def, one);
    var->store(ndef);

    return var;
}

const Def* PrefixExpr::remit(CodeGen& cg) const {
    if (kind() == ADD)
        return rhs()->remit(cg); // this is a NOP

    if (kind() == SUB) {
        const Def* def = rhs()->remit(cg);
        const anydsl::PrimType* pt = def->type()->as<anydsl::PrimType>();
        const anydsl::PrimLit* zero; 

        switch (pt->primtype_kind()) {
            case anydsl::PrimType_f32: zero = cg.world.literal_f32(-0.f); break;
            case anydsl::PrimType_f64: zero = cg.world.literal_f64(-0.0); break;
            default: 
                assert(pt->is_int()); 
                zero = cg.world.literal(pt->primtype_kind(), 0u);
        }

        return cg.world.arithop(anydsl::ArithOp_sub, zero, def);
    }

    return lemit(cg)->load();
}

Var* InfixExpr::lemit(CodeGen& cg) const {
    TokenKind op = (TokenKind) kind();
    assert(Token::isAsgn(op));

    const Id* id = lhs()->isa<Id>();

    // special case for 'a = expr' -> don't use lookup!
    Var* lvar = op == Token::ASGN && id
              ? cg.curBB->insert(id->symbol(), cg.world.bottom(cg.convert(id->type())))
              : lhs()->lemit(cg);

    const Def* ldef = lvar->load();
    const Def* rdef = rhs()->remit(cg);

    if (op != Token::ASGN) {
        TokenKind sop = Token::seperateAssign(op);
        rdef = cg.world.binop(Token::toBinOp(sop), ldef, rdef);
    }

    lvar->store(rdef);

    return lvar;
}

const Def* InfixExpr::remit(CodeGen& cg) const {
    TokenKind op = (TokenKind) kind();

    if (Token::isAsgn(op))
        return lemit(cg)->load();
        
    const Def* ldef = lhs()->remit(cg);
    const Def* rdef = rhs()->remit(cg);

    return cg.world.binop(Token::toBinOp(op), ldef, rdef);
}

Var* PostfixExpr::lemit(CodeGen& cg) const {
    Var* var = lhs()->lemit(cg);
    const Def* def = var->load();
    const anydsl::PrimType* pt = def->type()->as<anydsl::PrimType>();
    const anydsl::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
    const Def* ndef = cg.world.arithop(Token::toArithOp((TokenKind) kind()), def, one);
    var->store(ndef);

    return var;
}

const Def* PostfixExpr::remit(CodeGen& cg) const { return lemit(cg)->load(); }

//Ref IndexExpr::ref(CodeGen& cg) const {
    //const Literal* lit = index()->as<Literal>();
    //u32 pos = lit->box().get_u32();

    //return Ref(lhs()->lemit(cg), pos);
//}

Var* IndexExpr::lemit(CodeGen& cg) const {
    //if (const IndexExpr* iexpr = lhs->isa<IndexExpr>()) {
        //Ref ref = iexpr->ref(cg);
    //}

    Var* var = lhs()->lemit(cg);
    //const Def* pos = index()->remit(cg);

    //return cg.curBB->in
    return 0;
}

const Def* IndexExpr::remit(CodeGen& cg) const { 
    return cg.world.extract(lhs()->remit(cg), index()->as<Literal>()->box().get_u32());
}

Array<const Def*> Call::emit_ops(CodeGen& cg) const {
    size_t size = ops_.size();
    assert(size >= 1);
    Array<const Def*> ops(size);

    size_t i = 0;
    for_all (op, ops_)
        ops[i++] = op->remit(cg);

    return ops;
}

const Def* Call::remit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);
    const anydsl::Type* ret = cg.convert(type());

    if (ret)
        return cg.curBB->call(ops[0], ops.slice_back(1), ret);
    else {
        cg.curBB->tail_call(ops[0], ops.slice_back(1));
        cg.curBB = 0;
        return 0;
    }
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    Var* var = decl()->emit(cg);

    if (const Expr* init_expr = init())
        var->store(init_expr->remit(cg));
}

void ExprStmt::emit(CodeGen& cg) const {
    if (cg.reachable())
        expr()->remit(cg);
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
        const Def* c = cond()->remit(cg);
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
    const Def* c = cond()->remit(cg);
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
    const Def* c = cond()->remit(cg);
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
    const Def* c = cond()->remit(cg);
    headBB->branch(c, bodyBB, nextBB);
    bodyBB->seal();

    // body
    cg.curBB = bodyBB;
    body()->emit(cg);
    cg.fixto(stepBB);
    stepBB->seal();

    // step
    cg.curBB = stepBB;
    step()->remit(cg);
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
        const Def* def = expr()->remit(cg);
        cg.curBB->insert(anydsl::Symbol("<result>"), def);
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

const anydsl::Type* PrimType::convert(anydsl::World& world) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) \
        case TYPE_##itype: \
            return world.type_##atype();
#include "impala/tokenlist.h"
        default:
            ANYDSL_UNREACHABLE;
    }
}

const anydsl::Type* Void::convert(anydsl::World& world) const { return world.unit(); }
const anydsl::Type* NoRet::convert(anydsl::World& /*world*/) const { return 0; }
const anydsl::Type* TypeError::convert(anydsl::World& /*world*/) const { ANYDSL_UNREACHABLE; }

const anydsl::Type* Pi::convert(anydsl::World& world) const {
    size_t size = elems().size();
    Array<const anydsl::Type*> types(size + 1);

    for (size_t i = 0; i < size; ++i)
        types[i] = elems()[i]->convert(world);

    if (!ret()->is_noret())
        types[size++] = world.pi1(ret()->convert(world));

    return world.pi(types.slice_front(size));
}

const anydsl::Type* Sigma::convert(anydsl::World& world) const {
    size_t size = elems().size();
    Array<const anydsl::Type*> types(size);

    for (size_t i = 0; i < size; ++i)
        types[i] = elems()[i]->convert(world);

    return world.sigma(types);
}

} // namespace impala
