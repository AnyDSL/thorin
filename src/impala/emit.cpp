#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <iostream>

#include <cstdio> // TODO remove
#include <vector>

#include "anydsl2/irbuilder.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
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
        , cur_frame(0)
        , break_target(0)
        , continue_target(0)
    {}

    Lambda* cur_fun;
    const Def* cur_frame;

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
    lambda_ = cg.world().lambda(convert(pi()), symbol.str());
    size_t num = params().size();
    const Type* ret_type = return_type(pi());
    ret_param_ = ret_type->isa<NoRet>() ? 0 : lambda_->param(num-1+1);
    lambda()->param(0)->name = "mem";

    return lambda_;
}

const Lambda* Fun::emit_body(CodeGen& cg, Lambda* parent, const char* what) const {
    for_all (f, body()->named_funs())
        f->emit_head(cg, f->symbol());

    lambda()->set_parent(parent);

    Push<Lambda*> push1(cg.cur_bb,  lambda());
    Push<Lambda*> push2(cg.cur_fun, lambda());

    const Enter* enter = cg.world().enter(lambda()->param(0));
    cg.set_mem(enter->extract_mem());
    Push<const Def*> push3(cg.cur_frame, enter->extract_frame());

    size_t num = params().size();
    for (size_t i = 0; i < num; ++i) {
        const Param* p = lambda_->param(i+1);
        p->name = param(i)->symbol().str();
        param(i)->emit(cg)->store(p);
    }

    JumpTarget exit;
    body()->emit(cg, exit);
    cg.enter(exit);

    if (cg.is_reachable()) {
        if (is_continuation()) {
            std::cerr << what << " does not end with a call\n";
            cg.cur_bb->jump0(cg.world().bottom(cg.world().pi0()));
        } else {
            const Type* ret_type = return_type(pi());
            if (ret_type->isa<Void>())
                cg.cur_bb->jump1(ret_param(), cg.get_mem());
            else {
                std::cerr << what << " does not end with 'return'\n";
                cg.cur_bb->jump1(cg.world().bottom(cg.world().pi1(ret_type)), cg.get_mem());
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
    const Type* air_type = convert(type());
    if (is_address_taken())
        return Ref::create(cg.world().slot(air_type, handle(), cg.cur_frame, symbol().str()), cg);

    return Ref::create(cg.cur_bb, handle(), air_type, symbol().str());
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

    const VarDecl* vardecl = decl()->as<VarDecl>();
    const Type* air_type = convert(type());

    if (vardecl->is_address_taken())
        return Ref::create(cg.world().slot(air_type, vardecl->handle(), cg.cur_frame, symbol().str()), cg);

    return Ref::create(cg.cur_bb, vardecl->handle(), air_type, symbol().str());
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const PrimLit* one = cg.world().one(convert(def->type()));
            const Def* ndef = cg.world().arithop(Token::to_arithop((TokenKind) kind(), type()->is_float()), def, one);
            ref->store(ndef);
            return ref;
        }
        case ADD: return rhs()->emit(cg); // this is a NOP
        case SUB: return Ref::create(cg.world().arithop_minus(rhs()->emit(cg)->load()));
        case NOT:
        case L_N: return Ref::create(cg.world().arithop_not(rhs()->emit(cg)->load()));
        default: ANYDSL2_UNREACHABLE;
    }
}

RefPtr InfixExpr::emit(CodeGen& cg) const {
    const bool is_or = kind() == L_O;

    if (is_or || kind() == L_A) {
        JumpTarget t(is_or ? "l_or_true"  : "l_and_true");
        JumpTarget f(is_or ? "l_or_false" : "l_and_false");
        JumpTarget x(is_or ? "l_or_exit"  : "l_and_exit");
        lhs()->emit_branch(cg, t, f);

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

    const TokenKind op = (TokenKind) kind();

    if (Token::is_assign(op)) {
        //const Id* id = lhs()->isa<Id>();
        const Def* rdef = rhs()->emit(cg)->load();

        // special case for 'id = expr' -> don't use get_value!
        //RefPtr lref = op == Token::ASGN && id
                //? Ref::create(cg.cur_bb, id->decl()->as<VarDecl>()->handle(), convert(id->type()), id->symbol().str())
                //: lhs()->emit(cg);
        RefPtr lref = lhs()->emit(cg);

        if (op != Token::ASGN) {
            TokenKind sop = Token::separate_assign(op);
            rdef = cg.world().binop(Token::to_binop(sop, type()->is_float()), lref->load(), rdef);
        }

        lref->store(rdef);
        return lref;
    }
        
    const Def* ldef = lhs()->emit(cg)->load();
    const Def* rdef = rhs()->emit(cg)->load();

    return Ref::create(cg.world().binop(Token::to_binop(op, ldef->type()->is_float()), ldef, rdef));
}

RefPtr PostfixExpr::emit(CodeGen& cg) const {
    RefPtr ref = lhs()->emit(cg);
    const Def* def = ref->load();
    const PrimLit* one = cg.world().one(convert(def->type()));
    ref->store(cg.world().arithop(Token::to_arithop((TokenKind) kind(), type()->is_float()), def, one));
    return Ref::create(def);
}

RefPtr ConditionalExpr::emit(CodeGen& cg) const {
    JumpTarget t("cond_true");
    JumpTarget f("cond_false");
    JumpTarget x("cond_exit");

    cond()->emit_branch(cg, t, f);

    if (Lambda* tl = cg.enter(t)) {
        tl->set_value(0, t_expr()->emit(cg)->load());
        cg.jump(x);
    }

    if (Lambda* fl = cg.enter(f)) {
        fl->set_value(0, f_expr()->emit(cg)->load());
        cg.jump(x);
    }

    if (Lambda* xl = cg.enter(x))
        return Ref::create(xl->get_value(0, convert(t_expr()->type()), "cond"));
    return Ref::create(0);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    return Ref::create(lhs()->emit(cg), index()->emit(cg)->load());
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);
    Array<const Def*> args(num_args() + 1);
    std::copy(ops.begin() + 1, ops.end(), args.begin() + 1);
    args[0] = cg.get_mem();

    if (is_continuation_call()) {
        cg.cur_bb->jump(ops[0], args);
        cg.cur_bb = 0;
        return RefPtr(0);
    }

    RefPtr ref = Ref::create(cg.mem_call(ops[0], args, type()));
    cg.set_mem(cg.cur_bb->param(0));

    return ref;
}

/*
 * Expr -- emit_branch
 */

void Expr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const { cg.branch(emit(cg)->load(), t, f); }

void PrefixExpr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const {
    if (kind() == L_N)
        return rhs()->emit_branch(cg, f, t);
    cg.branch(emit(cg)->load(), t, f);
}

void InfixExpr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const {
    if (kind() == L_O || kind() == L_A) {
        bool is_or = kind() == L_O;
        JumpTarget extra(is_or ? "l_or_extra" : "l_and_extra");
        lhs()->emit_branch(cg, is_or ? t : extra, is_or ? extra : f);
        if (cg.enter(extra))
            rhs()->emit_branch(cg, t, f);
    } else
        Expr::emit_branch(cg, t, f);
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        RefPtr ref = var_decl()->emit(cg);
        if (const Expr* init_expr = init())
            ref->store(init_expr->emit(cg)->load());
        cg.jump(exit_bb);
    }
}

void ExprStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        expr()->emit(cg);
        cg.jump(exit_bb);
    }
}

void IfElseStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget then_bb("if_then");
    JumpTarget else_bb("if_else");

    cond()->emit_branch(cg, then_bb, else_bb);

    cg.enter(then_bb);
    then_stmt()->emit(cg, exit_bb);

    cg.enter(else_bb);
    else_stmt()->emit(cg, exit_bb);
}

void DoWhileStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget body_bb("do_while_body");
    JumpTarget cond_bb("do_while_cond");

    Push<JumpTarget*> push1(cg.break_target, &exit_bb);
    Push<JumpTarget*> push2(cg.continue_target, &cond_bb);

    cg.jump(body_bb);

    cg.enter_unsealed(body_bb);
    body()->emit(cg, cond_bb);

    cg.enter(cond_bb);
    cond()->emit_branch(cg, body_bb, exit_bb);
    body_bb.seal();
}

void ForStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget head_bb("for_head");
    JumpTarget body_bb("for_body");
    JumpTarget step_bb("for_step");

    Push<JumpTarget*> push1(cg.break_target, &exit_bb);
    Push<JumpTarget*> push2(cg.continue_target, &step_bb);

    init()->emit(cg, head_bb);

    cg.enter_unsealed(head_bb);
    cond()->emit_branch(cg, body_bb, exit_bb);

    cg.enter(body_bb);
    body()->emit(cg, step_bb);

    cg.enter(step_bb);
    step()->emit(cg);
    cg.jump(head_bb);
    head_bb.seal();
}

const Lambda* ForeachStmt::emit_ret(CodeGen& cg, Lambda* parent, JumpTarget& exit_bb) const {
    std::vector<const Type*> ret_elems;
    const Pi* ret_type = cg.world().pi(ret_elems);
    
    // emit head
    anydsl2::Lambda* lambda_ = cg.world().lambda(convert(ret_type), "foreach_ret");
    lambda_->param(0)->name = "mem";
    
    // emit body
    lambda_->set_parent(parent);

    Push<Lambda*> push1(cg.cur_bb,  lambda_);
    Push<Lambda*> push2(cg.cur_fun, lambda_);

    const Enter* enter = cg.world().enter(lambda_->param(0));
    cg.set_mem(enter->extract_mem());
    Push<const Def*> push3(cg.cur_frame, enter->extract_frame());

    JumpTarget exit;
    // body of the lambda: just a jump out of the foreach
    cg.jump(exit_bb);
    
    cg.jump(exit);
    cg.enter(exit);
    return lambda_;
}

void ForeachStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    size_t num = ops_.size();
    Array<const Def*> defs(num + 2);
    for (size_t i = 0; i < num; ++i)
        defs[i] = op(i)->emit(cg)->load();
    
    // TODO store value to lhs in each iteration
    // TODO call 'next'
    // lambda (..., step : pi()) { lhs = val; body; next(); }
    std::vector<const Type*> elems;
    elems.push_back(cg.world().type_u32()); // TODO this has to be the left hand side type
    std::vector<const Type*> inner_elems;
    elems.push_back(cg.world().pi(inner_elems));
    const Pi* pi = cg.world().pi(elems);
    FunExpr* fun = new FunExpr();
    if (const ScopeStmt* scope = body()->isa<ScopeStmt>())
        fun->fun_set(pi, scope);
    // else TODO ...
    defs[num] = fun->emit(cg)->load();

    // lambda() { jump exit_bb }
    defs[num + 1] = Ref::create(emit_ret(cg, cg.cur_bb, exit_bb))->load();
        
    Array<const Def*> ops = defs;
    Array<const Def*> args(num + 2);
    std::copy(ops.begin() + 1, ops.end(), args.begin() + 1);
    args[0] = cg.get_mem();
    RefPtr call_ref = Ref::create(cg.mem_call(ops[0], args, call_type()));
    cg.set_mem(cg.cur_bb->param(0));
    
    // TODO remove this hack
    RefPtr ref;
    if (init_decl()) {
        ref = init_decl()->emit(cg);
    } else {
        ref = init_expr()->emit(cg);
    }
    ref->store(call_ref->load());
    cg.jump(exit_bb);
}

void BreakStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.break_target); }
void ContinueStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.continue_target); }

void ReturnStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        const Param* ret_param = fun()->ret_param();
        if (const Call* call = expr()->isa<Call>()) {
            Array<const Def*> ops = call->emit_ops(cg);
            Array<const Def*> args(call->num_args() + 2);
            std::copy(ops.begin() + 1, ops.end(), args.begin() + 1);
            args[0] = cg.get_mem();
            args.back() = ret_param;
            cg.tail_call(ops[0], args);
        } else {
            if (expr()) 
                cg.return2(ret_param, cg.world().leave(cg.get_mem(), cg.cur_frame), expr()->emit(cg)->load());
            else
                cg.return1(ret_param, cg.world().leave(cg.get_mem(), cg.cur_frame));
        }
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
            stmt(i)->emit(cg, stmt_exit_bb);
            cg.enter(stmt_exit_bb);
        }
        stmt(i)->emit(cg, exit_bb);
    }
}

void NamedFunStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { 
    named_fun()->emit(cg); 
    cg.jump(exit_bb);
}

} // namespace impala
