#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <iostream>

#include "anydsl2/irbuilder.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/util/push.h"
#include "anydsl2/world.h"

#include "impala/type.h"

using namespace anydsl2;

namespace impala {

class CodeGen : public IRBuilder {
public:

    CodeGen(World& world)
        : IRBuilder(world)
        , cur_fun(0)
        , break_target(0)
        , continue_target(0)
    {}

    Lambda* cur_fun;
    JumpTarget* break_target;
    JumpTarget* continue_target;
};

//------------------------------------------------------------------------------

void emit(World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
}

//------------------------------------------------------------------------------

Lambda* Fun::emit_head(CodeGen& cg, Symbol symbol) const {
    lambda_ = cg.world().lambda(pi(), symbol.str());
    size_t num = params().size();
    const Type* ret_type = return_type(pi());
    ret_param_ = ret_type->isa<NoRet>() ? 0 : lambda_->param(num-1);

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

    Push<Lambda*> push1(cg.cur_bb,  lambda());
    Push<Lambda*> push2(cg.cur_fun, lambda());

    JumpTarget exit;
    body()->emit(cg, exit);
    cg.enter(exit);

    if (cg.reachable()) {
        if (is_continuation()) {
            std::cerr << what << " does not end with a call\n";
            cg.cur_bb->jump0(cg.world().bottom(cg.world().pi0()));
        } else {
            const Type* ret_type = return_type(pi());
            if (ret_type->isa<Void>())
                cg.cur_bb->jump0(ret_param());
            else {
                std::cerr << what << " does not end with 'return'\n";
                cg.cur_bb->jump0(cg.world().bottom(cg.world().pi1(ret_type)));
            }
        }
    }

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
    emit_body(cg, cg.cur_bb, symbol().str());
    if (extern_)
        lambda()->attr().set_extern();
}

RefPtr VarDecl::emit(CodeGen& cg) const {
    return Ref::create(cg.cur_bb, handle(), type(), symbol().str());
}

/*
 * Expr -- emit
 */

Array<const Def*> Expr::emit_ops(CodeGen& cg, size_t additional_size) const {
    size_t num = ops_.size();
    Array<const Def*> defs(num + additional_size);
    for (size_t i = 0; i < num; ++i)
        defs[i] = op(i)->emit(cg)->load();

    return defs;
}

RefPtr EmptyExpr::emit(CodeGen& cg) const { 
    return Ref::create(cg.world().bottom(cg.world().unit())); 
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

    return Ref::create(cg.world().literal(akind, box()));
}

RefPtr FunExpr::emit(CodeGen& cg) const {
    emit_head(cg, "lambda");
    return Ref::create(emit_body(cg, cg.cur_bb, "anonymous lambda expression"));
}

RefPtr Tuple::emit(CodeGen& cg) const {
    return Ref::create(cg.world().tuple(emit_ops(cg)));
}

RefPtr Id::emit(CodeGen& cg) const {
    if (const NamedFun* named_fun = decl()->isa<NamedFun>())
        return Ref::create(named_fun->lambda());
    return Ref::create(cg.cur_bb, decl()->as<VarDecl>()->handle(), type(), symbol().str());
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const PrimType* pt = def->type()->as<PrimType>();
            const PrimLit* one = cg.world().literal(pt->primtype_kind(), 1u);
            const Def* ndef = cg.world().arithop(Token::to_arithop((TokenKind) kind()), def, one);
            ref->store(ndef);

            return ref;
        }
        case ADD:
            return rhs()->emit(cg); // this is a NOP
        case SUB: {
            const Def* def = rhs()->emit(cg)->load();
            const PrimType* pt = def->type()->as<PrimType>();
            const PrimLit* zero; 

            switch (pt->primtype_kind()) {
                case PrimType_f32: zero = cg.world().literal_f32(-0.f); break;
                case PrimType_f64: zero = cg.world().literal_f64(-0.0); break;
                default: 
                    assert(pt->is_int()); 
                    zero = cg.world().literal(pt->primtype_kind(), 0u);
            }

            return Ref::create(cg.world().arithop(ArithOp_sub, zero, def));
        }
        case L_N:
            return Ref::create(cg.world().arithop_xor(rhs()->emit(cg)->load(), cg.world().literal_u1(true)));
        default: ANYDSL2_UNREACHABLE;
    }
}

RefPtr InfixExpr::emit(CodeGen& cg) const {
    TokenKind op = (TokenKind) kind();

    if (kind() == L_O || kind() == L_A) {
        const bool is_or = kind() == L_O;
        JumpTarget t(is_or ? "l_or_t" : "l_and_t");
        JumpTarget f(is_or ? "l_or_f" : "l_and_f");
        JumpTarget x(is_or ? "l_or_x" : "l_and_x");
        lhs()->emit_cf(cg, t, f);

        if (Lambda* tl = cg.enter(t)) {
            tl->set_value(0, is_or ? cg.world().literal_u1(true) : rhs()->emit(cg)->load());
            cg.jump(x);
        }

        if (Lambda* fl = cg.enter(f)) {
            fl->set_value(0, is_or ? rhs()->emit(cg)->load() : cg.world().literal_u1(false));
            cg.jump(x);
        }

        if (Lambda* xl = cg.enter(x))
            return Ref::create(xl->get_value(0, cg.world().type_u1(), is_or ? "l_or" : "l_and"));
        return Ref::create(0);
    }

    if (Token::is_asgn(op)) {
        const Id* id = lhs()->isa<Id>();
        const Def* rdef = rhs()->emit(cg)->load();

        // special case for 'id = expr' -> don't use get_value!
        RefPtr lref = op == Token::ASGN && id
                ? Ref::create(cg.cur_bb, id->decl()->as<VarDecl>()->handle(), id->type(), id->symbol().str())
                : lhs()->emit(cg);

        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            rdef = cg.world().binop(Token::to_binop(sop), lref->load(), rdef);
        }

        lref->store(rdef);
        return lref;
    }
        
    const Def* ldef = lhs()->emit(cg)->load();
    const Def* rdef = rhs()->emit(cg)->load();

    return Ref::create(cg.world().binop(Token::to_binop(op), ldef, rdef));
}

RefPtr PostfixExpr::emit(CodeGen& cg) const {
    RefPtr ref = lhs()->emit(cg);
    const Def* def = ref->load();
    const PrimType* pt = def->type()->as<PrimType>();
    const PrimLit* one = cg.world().literal(pt->primtype_kind(), 1u);
    ref->store(cg.world().arithop(Token::to_arithop((TokenKind) kind()), def, one));
    return Ref::create(def);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    return Ref::create(lhs()->emit(cg), index()->emit(cg)->load());
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);

    if (is_continuation_call()) {
        cg.cur_bb->jump(ops[0], ops.slice_back(1));
        cg.cur_bb = 0;
        return RefPtr(0);
    } else {
        cg.call(ops[0], ops.slice_back(1), type());
        return Ref::create(cg.cur_bb->param(0));
    }
}

/*
 * Expr -- emit_cf
 */

void Expr::emit_cf(CodeGen& cg, JumpTarget& t, JumpTarget& f) const {
    cg.branch(emit(cg)->load(), t, f);
}

void PrefixExpr::emit_cf(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const {
    if (kind() == L_N)
        return rhs()->emit_cf(cg, f, t);
    cg.branch(emit(cg)->load(), t, f);
}

void InfixExpr::emit_cf(CodeGen& cg, anydsl2::JumpTarget& t, anydsl2::JumpTarget& f) const {
    if (kind() == L_O || kind() == L_A) {
        bool is_or = kind() == L_O;
        JumpTarget extra(is_or ? "l_or_e" : "l_and_e");
        lhs()->emit_cf(cg, is_or ? t : extra, is_or ? extra : f);
        if (cg.enter(extra))
            rhs()->emit_cf(cg, t, f);
    } else
        Expr::emit_cf(cg, t, f);
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.reachable()) {
        RefPtr ref = var_decl()->emit(cg);
        if (const Expr* init_expr = init())
            ref->store(init_expr->emit(cg)->load());
        cg.jump(exit_bb);
    }
}

void ExprStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.reachable()) {
        expr()->emit(cg);
        cg.jump(exit_bb);
    }
}

void IfElseStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget then_bb("then");
    JumpTarget else_bb("else");

    cond()->emit_cf(cg, then_bb, else_bb);

    cg.enter(then_bb);
    then_stmt()->emit(cg, exit_bb);

    cg.enter(else_bb);
    else_stmt()->emit(cg, exit_bb);
}

void DoWhileStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget body_bb("do_while-body");
    JumpTarget cond_bb("do_while-cond");

    Push<JumpTarget*> push1(cg.break_target, &exit_bb);
    Push<JumpTarget*> push2(cg.continue_target, &cond_bb);

    cg.jump(body_bb);

    cg.enter_unsealed(body_bb);
    body()->emit(cg, cond_bb);

    cg.enter(cond_bb);
    cond()->emit_cf(cg, body_bb, exit_bb);
    body_bb.seal();
}

void ForStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget head_bb("for-head");
    JumpTarget body_bb("for-body");
    JumpTarget step_bb("for-step");

    Push<JumpTarget*> push1(cg.break_target, &exit_bb);
    Push<JumpTarget*> push2(cg.continue_target, &step_bb);

    init()->emit(cg, head_bb);

    cg.enter_unsealed(head_bb);
    cond()->emit_cf(cg, body_bb, exit_bb);

    cg.enter(body_bb);
    body()->emit(cg, step_bb);

    cg.enter(step_bb);
    step()->emit(cg);
    cg.jump(head_bb);
    head_bb.seal();
}

void BreakStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.break_target); }
void ContinueStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.continue_target); }

void ReturnStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.reachable()) {
        const Param* ret_param = fun()->ret_param();
        if (const Call* call = expr()->isa<Call>()) {
            Array<const Def*> ops = call->emit_ops(cg, 1);
            ops.back() = ret_param;
            cg.cur_bb->jump(ops[0], ops.slice_back(1));
        } else {
            if (expr()) 
                cg.cur_bb->jump1(ret_param, expr()->emit(cg)->load());
            else
                cg.cur_bb->jump0(ret_param); // return void
        }
        cg.cur_bb = 0;
    }
}

void ScopeStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    size_t size = stmts().size();
    if (size == 0)
        cg.jump(exit_bb);
    else {
        size_t i = 0;
        for (; i != size - 1; ++i) {
            JumpTarget stmt_exit_bb("next");
            stmts()[i]->emit(cg, stmt_exit_bb);
            cg.enter(stmt_exit_bb);
        }
        stmts()[i]->emit(cg, exit_bb);
    }
}

void NamedFunStmt::emit(CodeGen& cg, anydsl2::JumpTarget& exit_bb) const { 
    named_fun()->emit(cg); 
    cg.jump(exit_bb);
}

} // namespace impala
