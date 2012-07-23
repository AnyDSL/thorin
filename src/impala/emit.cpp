#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <boost/scoped_array.hpp>

#include "anydsl/cfg.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

#include "impala/type.h"

using anydsl::BB;
using anydsl::Def;
using anydsl::FctParam;
using anydsl::FctParams;
using anydsl::Var;
using anydsl::make_name;
using anydsl::World;

namespace impala {

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

    anydsl::BB* curBB;
    anydsl::Fct* curFct;
    anydsl::World& world;
    boost::unordered_map<anydsl::Symbol, anydsl::Fct*> fcts;
};

//------------------------------------------------------------------------------

void emit(anydsl::World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
    cg.world.cleanup();
}

//------------------------------------------------------------------------------

void Prg::emit(CodeGen& cg) const {
    for_all (f, fcts()) {
        FctParams fparams;

        for_all (param, f->params())
            fparams.push_back(FctParam(param->symbol(), param->type()->emit(cg.world)));

        const anydsl::Type* retType = f->pi()->retType()->emit(cg.world);
        cg.fcts[f->symbol()] = new anydsl::Fct(cg.world, fparams, retType, f->symbol().str());
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
    cg.fixto(cg.curFct->exit());

    cg.curFct->emit();

    cg.curFct = 0;
    cg.curBB = 0;
}

Var* Decl::emit(CodeGen& cg) const {
    Var* var = cg.curBB->setVar(symbol(), cg.world.undef(type()->emit(cg.world)));
    var->def->debug = symbol().str();

    return var;
}

/*
 * Expr
 */

Value EmptyExpr::emit(CodeGen& cg) const {
    return Value((const Def*) 0);
}

Value Literal::emit(CodeGen& cg) const {
    anydsl::PrimLitKind akind;

    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: akind = anydsl::PrimLit_##atype; break;
#include "impala/tokenlist.h"
        case LIT_bool: akind = anydsl::PrimLit_u1; break;
        default: ANYDSL_UNREACHABLE;
    }

    return Value(cg.world.literal(akind, value()));
}

Value Id::emit(CodeGen& cg) const {
    if (type()->isa<Pi>()) {
        anydsl_assert(cg.fcts.find(symbol()) != cg.fcts.end(), "function not found");
        return Value(cg.fcts[symbol()]->topLambda());
    }

    return Value(cg.curBB->getVar(symbol(), type()->emit(cg.world)));
}

Value PrefixExpr::emit(CodeGen& cg) const {
    Value val = rhs()->emit(cg);
    const Def* def = val.load();
    const anydsl::PrimType* p = def->type()->as<anydsl::PrimType>();

    if (kind() == ADD)
        return def; // this is a NOP

    switch (kind()) {
        case SUB: {
            // TODO incorrect for f32, f64
            const anydsl::PrimLit* zero = cg.world.literal(p->kind(), 0u);
            return cg.world.arithOp(anydsl::ArithOp_sub, zero, def);
        }
        case INC:
        case DEC: {
            const anydsl::PrimLit* one = cg.world.literal(p->kind(), 1u);
            const Def* ndef = cg.world.arithOp(Token::toArithOp((TokenKind) kind()), def, one);
            val.store(ndef);

            return val;
        }
        default: ANYDSL_UNREACHABLE; // TODO
    }
}

static Value emitInfix(anydsl::World& world, TokenKind op, Value vlhs, Value vrhs) {
    if (Token::isAsgn(op)) {
        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            vrhs = emitInfix(world, sop, vlhs, vrhs);
        }

        vlhs.store(vrhs.load());

        return vlhs;
    }

    if (Token::isArith(op))
        return world.arithOp(Token::toArithOp(op), vlhs.load(), vrhs.load());

    anydsl_assert(Token::isRel(op), "must be a relop");
    return world.relOp(Token::toRelOp(op), vlhs.load(), vrhs.load());
}

Value InfixExpr::emit(CodeGen& cg) const {
    Value vlhs = lhs()->emit(cg);
    Value vrhs = rhs()->emit(cg);

    return emitInfix(cg.world, (TokenKind) kind(), vlhs, vrhs);
}

Value PostfixExpr::emit(CodeGen& cg) const {
    Value val = lhs()->emit(cg);
    const Def* def = val.load();
    const anydsl::PrimType* p = def->type()->as<anydsl::PrimType>();

    const anydsl::PrimLit* one = cg.world.literal(p->kind(), 1u);
    const Def* ndef = cg.world.arithOp(Token::toArithOp((TokenKind) kind()), def, one);
    val.store(ndef);

    return Value(def);
}

Value Call::emit(CodeGen& cg) const {
    size_t size = args_.size();
    assert(size >= 1);
    boost::scoped_array<const Def*> args(new const Def*[size]);

    size_t i = 0;
    for_all (arg, args_) {
        args[i] = arg->emit(cg).load();
        ++i;
    }

    const anydsl::Type* retType = type()->emit(cg.world);

    return Value(cg.curBB->calls(args[0], args.get() + 1, args.get() + size, retType));
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    Var* var = decl()->emit(cg);

    if (init()) {
        const Def* def = init()->emit(cg).load();
        var->def = def;
    }
}

void ExprStmt::emit(CodeGen& cg) const {
    if (cg.reachable())
        expr()->emit(cg);
}

void IfElseStmt::emit(CodeGen& cg) const {
    static int id = 0;

    // always create elseBB -- the edge from headBB to nextBB would be crtical anyway

    // create BBs
    BB* thenBB = cg.curFct->createBB(make_name("if-then", id));
    BB* elseBB = cg.curFct->createBB(make_name("if-else", id));

    // condition
    if (cg.reachable()) {
        const Def* cvar = cond()->emit(cg).load();
        cg.curBB->branches(cvar, thenBB, elseBB);
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

    if (thenCur && elseCur) {
        BB* nextBB = cg.curFct->createBB(make_name("if-next", id));
        thenCur->fixto(nextBB);
        elseCur->fixto(nextBB);
        nextBB->seal();
    } else if (thenCur || elseCur)
        cg.curBB = thenCur ? thenCur : elseCur;
    else 
        cg.curBB = 0;

    ++id;
}

void WhileStmt::emit(CodeGen& cg) const {
    static int id = 0;

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("while-head", id));
    BB* bodyBB = cg.curFct->createBB(make_name("while-body", id));
    BB* nextBB = cg.curFct->createBB(make_name("while-next", id));

    // head
    cg.fixto(headBB);
    cg.curBB = headBB;

    // condition
    const Def* cvar = cond()->emit(cg).load();
    headBB->branches(cvar, bodyBB, nextBB);
    bodyBB->seal();

    // body
    cg.curBB = bodyBB;
    body()->emit(cg);
    cg.fixto(headBB);
    headBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();

    ++id;
}

void DoWhileStmt::emit(CodeGen& cg) const {
    body()->emit(cg);
    cond()->emit(cg);
}

void ForStmt::emit(CodeGen& cg) const {
    static int id = 0;

    init()->emit(cg);

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("for-head", id));
    BB* bodyBB = cg.curFct->createBB(make_name("for-body", id));
    BB* stepBB = cg.curFct->createBB(make_name("for-step", id));
    BB* nextBB = cg.curFct->createBB(make_name("for-next", id));

    // head
    cg.fixto(headBB);
    cg.curBB = headBB;

    // condition
    const Def* cvar = cond()->emit(cg).load();
    headBB->branches(cvar, bodyBB, nextBB);
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

    ++id;
}

void BreakStmt::emit(CodeGen& cg) const {
}

void ContinueStmt::emit(CodeGen& cg) const {
}

void ReturnStmt::emit(CodeGen& cg) const {
    const Def* def = expr()->emit(cg).load();

    cg.curBB->setVar(anydsl::Symbol("<result>"), def);
    cg.curBB->goesto(cg.curFct->exit());
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

const anydsl::Type* PrimType::emit(anydsl::World& world) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) \
        case TYPE_##itype: \
            return world.type_##atype();
#include "impala/tokenlist.h"
        default:
            ANYDSL_UNREACHABLE;
    }
}

const anydsl::Type* Void::emit(anydsl::World& world) const {
    return world.unit();
}

const anydsl::Type* NoRet::emit(anydsl::World& /*world*/) const {
    ANYDSL_UNREACHABLE;
}

const anydsl::Type* TypeError::emit(anydsl::World& /*world*/) const {
    ANYDSL_UNREACHABLE;
}

const anydsl::Type* Pi::emit(anydsl::World& world) const {
    std::vector<const anydsl::Type*> types;

    for (size_t i = 0; i < numArgs(); ++i)
        types.push_back(args()[i]->emit(world));

    if (!retType()->isNoRet())
        types.push_back(world.pi1(retType()->emit(world)));

    return world.pi(types.begin().base(), types.end().base());
}

} // namespace impala
