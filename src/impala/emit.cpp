#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <iostream>

#include "anydsl2/jumptarget.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/ref.h"
#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/world.h"

#include "impala/type.h"

using namespace anydsl2;

namespace impala {

class CodeGen {
public:

    CodeGen(World& world)
        : world(world)
        , curBB(0)
        , curFun(0)
    {}

    Lambda* basicblock(const char* name) { return world.basicblock(name); }
    bool reachable() const { return curBB; }
    void fixto(Lambda* to) { if (reachable()) curBB->jump0(to); }
    void fixto(JumpTarget& jt) { if (reachable()) curBB->jump(jt); }
    void fixto(Lambda* from, Lambda* to) { if (from) from->jump0(to); }

    World& world;
    Lambda* curBB;
    Lambda* curFun;
};

//------------------------------------------------------------------------------

void emit(World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
    cg.world.cleanup();
}

//------------------------------------------------------------------------------

Lambda* Fun::emit_head(CodeGen& cg, Symbol symbol) const {
    lambda_ = cg.world.lambda(pi(), symbol.str());
    size_t num = params().size();
    const Type* ret_type = return_type(pi());
    if (!ret_type->isa<NoRet>()) {
        ret_param_ = lambda_->param(num-1);
    } else
        ret_param_ = 0;

    for (size_t i = 0; i < num; ++i) {
        const Param* p = lambda_->param(i);
        p->name = param(i)->symbol().str();
        lambda_->set_value(param(i)->handle(), lambda_->param(i));
    }

    return lambda_;
}

const Lambda* Fun::emit_body(CodeGen& cg, Lambda* parent, const char* what) const {
    for_all (f, body()->named_funs())
        f->emit_head(cg, f->symbol());

    lambda()->set_parent(parent);
    Lambda* oldBB = cg.curBB;
    Lambda* oldFun = cg.curFun;
    cg.curBB = cg.curFun = lambda();

    body()->emit(cg);

    if (cg.reachable()) {
        if (is_continuation()) {
            std::cerr << what << " does not end with a call\n";
            cg.curBB->jump0(cg.world.bottom(cg.world.pi0()));
        } else {
            const Type* ret_type = return_type(pi());
            if (ret_type->isa<Void>())
                cg.curBB->jump0(ret_param());
            else {
                std::cerr << what << " does not end with 'return'\n";
                cg.curBB->jump0(cg.world.bottom(cg.world.pi1(ret_type)));
            }
        }
    }

    cg.curBB  = oldBB;
    cg.curFun = oldFun;

    return lambda();
}

//------------------------------------------------------------------------------

void Prg::emit(CodeGen& cg) const {
    for_all (f, named_funs()) {
        Lambda* lambda = f->emit_head(cg, f->symbol());

        if (f->symbol() == Symbol("main"))
            lambda->attr().set_extern();
    }

    for_all (f, named_funs())
        f->emit(cg);
}

void NamedFun::emit(CodeGen& cg) const {
    emit_body(cg, cg.curBB, symbol().str());
    if (extern_)
        lambda()->attr().set_extern();
}

RefPtr VarDecl::emit(CodeGen& cg) const {
    return Ref::create(cg.curBB, handle(), type(), symbol().str());
}

/*
 * Expr
 */

Array<const Def*> Expr::emit_ops(CodeGen& cg, size_t additional_size) const {
    size_t num = ops_.size();
    Array<const Def*> defs(num + additional_size);
    for (size_t i = 0; i < num; ++i)
        defs[i] = op(i)->emit(cg)->load();

    return defs;
}

RefPtr EmptyExpr::emit(CodeGen& cg) const { 
    return Ref::create(cg.world.bottom(cg.world.unit())); 
}

RefPtr Literal::emit(CodeGen& cg) const {
    PrimTypeKind akind;

    switch (kind()) {
#define IMPALA_LIT(itype, atype) \
        case LIT_##itype: akind = PrimType_##atype; break;
#include "impala/tokenlist.h"
        case LIT_bool: akind = PrimType_u1; break;
        default: ANYDSL2_UNREACHABLE;
    }

    return Ref::create(cg.world.literal(akind, box()));
}

RefPtr FunExpr::emit(CodeGen& cg) const {
    emit_head(cg, "lambda");
    return Ref::create(emit_body(cg, cg.curBB, "anonymous lambda expression"));
}

RefPtr Tuple::emit(CodeGen& cg) const {
    return Ref::create(cg.world.tuple(emit_ops(cg)));
}

RefPtr Id::emit(CodeGen& cg) const {
    if (const NamedFun* named_fun = decl()->isa<NamedFun>())
        return Ref::create(named_fun->lambda());
    return Ref::create(cg.curBB, decl()->as<VarDecl>()->handle(), type(), symbol().str());
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const PrimType* pt = def->type()->as<PrimType>();
            const PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
            const Def* ndef = cg.world.arithop(Token::to_arithop((TokenKind) kind()), def, one);
            ref->store(ndef);

            return ref;
        }
        case ADD:
            return rhs()->emit(cg); // this is a NOP

        case SUB: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const PrimType* pt = def->type()->as<PrimType>();
            const PrimLit* zero; 

            switch (pt->primtype_kind()) {
                case PrimType_f32: zero = cg.world.literal_f32(-0.f); break;
                case PrimType_f64: zero = cg.world.literal_f64(-0.0); break;
                default: 
                    assert(pt->is_int()); 
                    zero = cg.world.literal(pt->primtype_kind(), 0u);
            }

            return Ref::create(cg.world.arithop(ArithOp_sub, zero, def));
        }
        default: ANYDSL2_UNREACHABLE;
    }
}

RefPtr InfixExpr::emit(CodeGen& cg) const {
    TokenKind op = (TokenKind) kind();

    if (Token::is_asgn(op)) {
        const Id* id = lhs()->isa<Id>();
        const Def* rdef = rhs()->emit(cg)->load();

        // special case for 'id = expr' -> don't use get_value!
        RefPtr lref = op == Token::ASGN && id
                ? Ref::create(cg.curBB, id->decl()->as<VarDecl>()->handle(), id->type(), id->symbol().str())
                : lhs()->emit(cg);

        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            rdef = cg.world.binop(Token::to_binop(sop), lref->load(), rdef);
        }

        lref->store(rdef);
        return lref;
    }

    const Def* ldef = lhs()->emit(cg)->load();
    const Def* rdef = rhs()->emit(cg)->load();

    return Ref::create(cg.world.binop(Token::to_binop(op), ldef, rdef));
}

RefPtr PostfixExpr::emit(CodeGen& cg) const {
    RefPtr ref = lhs()->emit(cg);
    const Def* def = ref->load();
    const PrimType* pt = def->type()->as<PrimType>();
    const PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
    ref->store(cg.world.arithop(Token::to_arithop((TokenKind) kind()), def, one));
    return Ref::create(def);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    return Ref::create(lhs()->emit(cg), index()->emit(cg)->load());
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);

    if (is_continuation_call()) {
        cg.curBB->jump(ops[0], ops.slice_back(1));
        cg.curBB = 0;
        return RefPtr(0);
    } else {
        cg.curBB = cg.curBB->call(ops[0], ops.slice_back(1), type());
        return Ref::create(cg.curBB->param(0));
    }
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    if (cg.reachable()) {
        RefPtr ref = var_decl()->emit(cg);
        if (const Expr* init_expr = init())
            ref->store(init_expr->emit(cg)->load());
    }
}

void ExprStmt::emit(CodeGen& cg) const {
    if (cg.reachable())
        expr()->emit(cg);
}

void IfElseStmt::emit(CodeGen& cg) const {
    Lambda* thenBB = cg.basicblock("if-then");
    Lambda* elseBB = cg.basicblock("if-else");// always create -- the edge from headBB to nextBB is crtical

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
    Lambda* thenCur = cg.curBB;

    // else
    cg.curBB = elseBB;
    elseStmt()->emit(cg);
    Lambda* elseCur = cg.curBB;

    if (!elseCur) {
        cg.curBB = thenCur;
    } else if (thenCur) {
        Lambda* nextBB = cg.basicblock("if-next");
        cg.fixto(thenCur, nextBB);
        cg.fixto(elseCur, nextBB);
        nextBB->seal();
        cg.curBB = nextBB;
    }
}

void DoWhileStmt::emit(CodeGen& cg) const {
    Lambda* bodyBB = cg.basicblock("dowhile-body");
    Lambda* condBB = cg.basicblock("dowhile-cond");
    Lambda* critBB = cg.basicblock("dowhile-crit");
    Lambda* nextBB = cg.basicblock("dowhile-next");

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
    critBB->jump0(bodyBB);
    bodyBB->seal();

    // next
    cg.curBB = nextBB;
    nextBB->seal();
}

void ForStmt::emit(CodeGen& cg) const {
    init()->emit(cg);

    Lambda* headBB = cg.basicblock("for-head");
    Lambda* bodyBB = cg.basicblock("for-body");
    JumpTarget stepBB = JumpTarget("for-step");
    Lambda* nextBB = cg.basicblock("for-next");

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

    // step
    cg.curBB = stepBB.enter();
    step()->emit(cg);
    cg.fixto(headBB);
    headBB->seal();

    // next
    nextBB->seal();
    cg.curBB = nextBB->num_uses() ? nextBB : 0;
}

void BreakStmt::emit(CodeGen& cg) const {
}

void ContinueStmt::emit(CodeGen& cg) const {
}

void ReturnStmt::emit(CodeGen& cg) const {
    if (cg.reachable()) {
        const Param* ret_param = fun()->ret_param();
        if (const Call* call = expr()->isa<Call>()) {
            Array<const Def*> ops = call->emit_ops(cg, 1);
            ops.back() = ret_param;
            cg.curBB->jump(ops[0], ops.slice_back(1));
        } else {
            if (expr()) 
                cg.curBB->jump1(ret_param, expr()->emit(cg)->load());
            else
                cg.curBB->jump0(ret_param); // return void
        }

        // all other statements in the same BB are unreachable
        cg.curBB = 0;
    }
}


void ScopeStmt::emit(CodeGen& cg) const {
    for_all (const &s, stmts())
        s->emit(cg);
}

} // namespace impala
