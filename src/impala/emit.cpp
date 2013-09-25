#include "impala/ast.h"

#include <iostream>
#include <vector>

#include "anydsl2/irbuilder.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/push.h"
#include "anydsl2/world.h"

#include "impala/type.h"

using namespace anydsl2;

namespace impala {

class CodeGen : public IRBuilder {
public:
    CodeGen(World& world)
        : IRBuilder(world)
        , break_target(nullptr)
        , continue_target(nullptr)
    {}

    void emit_prg(const Scope*);
    void emit(const Scope*, JumpTarget&);
    Lambda* emit_head(const Fun* fun);
    const Lambda* emit_body(const Fun* fun);
    RefPtr emit(const VarDecl*);
    Array<const Def*> emit_ops(const Expr* expr, size_t additional_size = 0);
    RefPtr emit(const Expr* expr) { return is_reachable() ? expr->emit(*this) : nullptr; }
    void emit_branch(const Expr* expr, JumpTarget& t, JumpTarget& f) { expr->emit_branch(*this, t, f); }
    void emit(const Stmt* stmt, JumpTarget& exit) { if (is_reachable()) stmt->emit(*this, exit); }
    void emit(const Item* item) { item->emit(*this); }

    JumpTarget* break_target;
    JumpTarget* continue_target;
};

//------------------------------------------------------------------------------

void emit(World& world, const Scope* prg) { CodeGen(world).emit_prg(prg); }

//------------------------------------------------------------------------------

void CodeGen::emit_prg(const Scope* prg) {
    for (auto stmt : prg->stmts()) {
        auto item = stmt->as<ItemStmt>()->item();
        if (auto fun_item = item->isa<FunItem>()) {
            auto fun = fun_item->fun();
            Lambda* lambda = emit_head(fun);
            if (fun->symbol() == Symbol("main")) {
                lambda->name += "_impala";
                lambda->attr().set_extern();
            }
        } 
        //else if (auto decl_stmt = stmt->isa<DeclStmt>()) {
            //if (auto proto = decl_stmt->decl()->isa<Proto>()) {
                //auto lambda = world().lambda(proto->refined_type()->convert(world())->as<Pi>());
                //lambda->attr().set_extern();
                //lambda->name = proto->symbol().str();
            //}
        //}
    }

    for (auto stmt : prg->stmts())
        emit(stmt->as<ItemStmt>()->item());

    // clear get/set value stuff
    for (auto lambda : world().lambdas())
        lambda->clear();
}


void CodeGen::emit(const Scope* scope, JumpTarget& exit_bb) {
    for (auto stmt : scope->stmts()) {
        if (auto item_stmt = stmt->isa<ItemStmt>()) {
            if (auto fun_item = item_stmt->item()->isa<FunItem>())
                emit_head(fun_item->fun());
        }
    }

    size_t size = scope->stmts().size();
    if (size == 0)
        jump(exit_bb);
    else {
        size_t i = 0;
        for (; i != size - 1; ++i) {
            // TODO: fixme
            if (auto stmt = scope->stmt(i)->isa<Stmt>()) {
                JumpTarget stmt_exit_bb("next");
                emit(stmt, stmt_exit_bb);
                enter(stmt_exit_bb);
            }
        }
        // TODO: fixme
        if (auto stmt = scope->stmt(i)->isa<Stmt>())
            emit(stmt, exit_bb);
    }
}

Lambda* CodeGen::emit_head(const Fun* fun) {
    auto type = fun->refined_fntype();
    fun->lambda_ = world().lambda(type->convert(world())->as<Pi>(), fun->symbol().str());
    if (fun->is_extern())
        fun->lambda_->attr().set_extern();
    size_t num = fun->params().size();
    const Type* ret_type = type->return_type();
    fun->ret_param_ = ret_type->isa<NoRet>() ? nullptr : fun->lambda_->param(num-1+1);
    fun->lambda()->param(0)->name = "mem";

    return fun->lambda_;
}

const Lambda* CodeGen::emit_body(const Fun* fun) {
    fun->lambda()->set_parent(cur_bb);
    ANYDSL2_PUSH(cur_bb, fun->lambda());

    const Enter* enter_op = world().enter(fun->lambda()->param(0));
    set_mem(enter_op->extract_mem());
    fun->frame_ = enter_op->extract_frame();

    size_t num = fun->params().size();
    for (size_t i = 0; i < num; ++i) {
        const Param* p = fun->lambda_->param(i+1);
        p->name = fun->param(i)->symbol().str();
        emit(fun->param(i))->store(p);
    }

    JumpTarget exit;
    emit(fun->body(), exit);
    enter(exit);

    if (is_reachable()) {
        if (!fun->is_continuation() && fun->refined_fntype()->return_type()->is_void())
            cur_bb->jump1(fun->ret_param(), get_mem());
        else {
            if (fun->is_continuation())
                std::cerr << fun->symbol() << " does not end with a call\n";
            else
                std::cerr << fun->symbol() << " does not end with 'return'\n";
            cur_bb->jump0(world().bottom(world().pi0()));
        }
    }

    return fun->lambda();
}

//------------------------------------------------------------------------------

RefPtr CodeGen::emit(const VarDecl* decl) {
    const anydsl2::Type* air_type = decl->refined_type()->convert(world());
    if (decl->is_address_taken())
        return Ref::create(world().slot(air_type, decl->fun()->frame(), decl->handle(), decl->symbol().str()), *this);

    return Ref::create(cur_bb, decl->handle(), air_type, decl->symbol().str());
}

/*
 * Expr -- emit
 */

Array<const Def*> CodeGen::emit_ops(const Expr* expr, size_t additional_size) {
    size_t num = expr->ops().size();
    Array<const Def*> defs(num + additional_size);
    for (size_t i = 0; i < num; ++i)
        defs[i] = emit(expr->op(i))->load();

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
    cg.emit_head(fun());
    return Ref::create(cg.emit_body(fun()));
}

RefPtr Tuple::emit(CodeGen& cg) const {
    return Ref::create(cg.world().tuple(cg.emit_ops(this)));
}

RefPtr Id::emit(CodeGen& cg) const {
    if (auto fun = decl()->isa<Fun>())
        return Ref::create(fun->lambda());
    if (auto proto = decl()->isa<Proto>())
        return Ref::create(cg.world().lambda(proto->refined_fntype()->convert(cg.world())->as<Pi>(), 
                    LambdaAttr(LambdaAttr::Extern), proto->symbol().str()));

    auto vardecl = decl()->as<VarDecl>();
    auto air_type = type()->convert(cg.world());

    if (vardecl->is_address_taken())
        return Ref::create(cg.world().slot(air_type, vardecl->fun()->frame(), vardecl->handle(), symbol().str()), cg);

    return Ref::create(cg.cur_bb, vardecl->handle(), air_type, symbol().str());
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = cg.emit(rhs());
            const Def* def = ref->load();
            const Def* one = cg.world().one(def->type());
            const Def* ndef = cg.world().arithop(Token::to_arithop((TokenKind) kind(), type()->is_float()), def, one);
            ref->store(ndef);
            return ref;
        }
        case ADD: return cg.emit(rhs()); // this is a NOP
        case SUB: return Ref::create(cg.world().arithop_minus(cg.emit(rhs())->load()));
        case NOT:
        case L_N: return Ref::create(cg.world().arithop_not(cg.emit(rhs())->load()));
        default: ANYDSL2_UNREACHABLE;
    }
}

RefPtr InfixExpr::emit(CodeGen& cg) const {
    const bool is_or = kind() == L_O;

    if (is_or || kind() == L_A) {
        JumpTarget t(is_or ? "l_or_true"  : "l_and_true");
        JumpTarget f(is_or ? "l_or_false" : "l_and_false");
        JumpTarget x(is_or ? "l_or_exit"  : "l_and_exit");
        cg.emit_branch(lhs(), t, f);

        if (Lambda* tl = cg.enter(t)) {
            tl->set_value(0, is_or ? cg.world().literal_u1(true) : cg.emit(rhs())->load());
            cg.jump(x);
        }

        if (Lambda* fl = cg.enter(f)) {
            fl->set_value(0, is_or ? cg.emit(rhs())->load() : cg.world().literal_u1(false));
            cg.jump(x);
        }

        if (Lambda* xl = cg.enter(x))
            return Ref::create(xl->get_value(0, cg.world().type_u1(), is_or ? "l_or" : "l_and"));
        return Ref::create(nullptr);
    }

    const TokenKind op = (TokenKind) kind();

    if (Token::is_assign(op)) {
        const Def* rdef = cg.emit(rhs())->load();
        RefPtr lref = cg.emit(lhs());

        if (op != Token::ASGN) {
            TokenKind sop = Token::separate_assign(op);
            rdef = cg.world().binop(Token::to_binop(sop, type()->is_float()), lref->load(), rdef);
        }

        lref->store(rdef);
        return lref;
    }
        
    const Def* ldef = cg.emit(lhs())->load();
    const Def* rdef = cg.emit(rhs())->load();

    return Ref::create(cg.world().binop(Token::to_binop(op, ldef->type()->is_float()), ldef, rdef));
}

RefPtr PostfixExpr::emit(CodeGen& cg) const {
    RefPtr ref = cg.emit(lhs());
    const Def* def = ref->load();
    const Def* one = cg.world().one(def->type());
    ref->store(cg.world().arithop(Token::to_arithop((TokenKind) kind(), type()->is_float()), def, one));
    return Ref::create(def);
}

RefPtr ConditionalExpr::emit(CodeGen& cg) const {
    JumpTarget t("cond_true");
    JumpTarget f("cond_false");
    JumpTarget x("cond_exit");

    cg.emit_branch(cond(), t, f);

    if (Lambda* tl = cg.enter(t)) {
        tl->set_value(0, cg.emit(t_expr())->load());
        cg.jump(x);
    }

    if (Lambda* fl = cg.enter(f)) {
        fl->set_value(0, cg.emit(f_expr())->load());
        cg.jump(x);
    }

    if (Lambda* xl = cg.enter(x))
        return Ref::create(xl->get_value(0, t_expr()->type()->convert(cg.world()), "cond"));
    return Ref::create(nullptr);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    const Def* x = cg.emit(index())->load();
    return Ref::create(cg.emit(lhs()), x);
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = cg.emit_ops(this);
    Array<const Def*> args(num_args() + 1);
    std::copy(ops.begin() + 1, ops.end(), args.begin() + 1);
    args[0] = cg.get_mem();

    if (is_continuation_call()) {
        cg.cur_bb->jump(ops[0], args);
        cg.cur_bb = nullptr;
        return RefPtr(nullptr);
    }

    cg.mem_call(ops[0], args, type()->convert(cg.world()));
    cg.set_mem(cg.cur_bb->param(0));

    return Ref::create(cg.cur_bb->param(1));
}

/*
 * Expr -- emit_branch
 */

void Expr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const { cg.branch(cg.emit(this)->load(), t, f); }

void PrefixExpr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const {
    if (kind() == L_N)
        return cg.emit_branch(rhs(), f, t);
    cg.branch(emit(cg)->load(), t, f);
}

void InfixExpr::emit_branch(CodeGen& cg, JumpTarget& t, JumpTarget& f) const {
    if (kind() == L_O || kind() == L_A) {
        bool is_or = kind() == L_O;
        JumpTarget extra(is_or ? "l_or_extra" : "l_and_extra");
        cg.emit_branch(lhs(), is_or ? t : extra, is_or ? extra : f);
        if (cg.enter(extra))
            cg.emit_branch(rhs(), t, f);
    } else
        Expr::emit_branch(cg, t, f);
}

/*
 * Stmt
 */

void ItemStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.emit(item()); cg.jump(exit_bb); }

void InitStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        RefPtr ref = cg.emit(var_decl());
        if (init())
            ref->store(cg.emit(init())->load());
        cg.jump(exit_bb);
    }
}

void ExprStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        cg.emit(expr());
        cg.jump(exit_bb);
    }
}

void IfElseStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget then_bb("if_then");
    JumpTarget else_bb("if_else");

    cg.emit_branch(cond(), then_scope()->empty() ? exit_bb : then_bb, 
                           else_scope()->empty() ? exit_bb : else_bb);

    if (!then_scope()->empty()) {
        cg.enter(then_bb);
        cg.emit(then_scope(), exit_bb);
    }

    if (!else_scope()->empty()) {
        cg.enter(else_bb);
        cg.emit(else_scope(), exit_bb);
    }
}

void DoWhileStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget body_bb("do_while_body");
    JumpTarget cond_bb("do_while_cond");

    ANYDSL2_PUSH(cg.break_target, &exit_bb);
    ANYDSL2_PUSH(cg.continue_target, &cond_bb);

    cg.jump(body_bb);

    cg.enter_unsealed(body_bb);
    cg.emit(body(), cond_bb);

    cg.enter(cond_bb);
    cg.emit_branch(cond(), body_bb, exit_bb);
    body_bb.seal();
}

void ForStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget head_bb("for_head");
    JumpTarget body_bb("for_body");
    JumpTarget step_bb("for_step");

    ANYDSL2_PUSH(cg.break_target, &exit_bb);
    ANYDSL2_PUSH(cg.continue_target, &step_bb);

    cg.emit(init(), head_bb);

    cg.enter_unsealed(head_bb);
    cg.emit_branch(cond(), body_bb, exit_bb);

    cg.enter(body_bb);
    cg.emit(body(), step_bb);

    cg.enter(step_bb);
    cg.emit(step());
    cg.jump(head_bb);
    head_bb.seal();
}

void ForeachStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    JumpTarget continue_bb("continue_bb");
    auto fun = fun_expr()->fun();

    ANYDSL2_PUSH(cg.break_target, &exit_bb);
    ANYDSL2_PUSH(cg.continue_target, &continue_bb);

    // construct a call to the generator
    size_t num = call()->ops().size();
    Array<const Def*> args(num + 1);
    args[0] = cg.get_mem();
    for (size_t i = 1; i < num; ++i)
        args[i] = cg.emit(call()->op(i))->load();

    Lambda* lambda = cg.emit_head(fun);
    args[num] = lambda;

    cg.mem_call(cg.emit(call()->op(0))->load(), args, nullptr /*no return type*/);
    Lambda* next = cg.cur_bb;
    cg.set_mem(cg.cur_bb->param(0));

    lambda->set_parent(cg.cur_bb);
    cg.cur_bb = lambda;
    cg.set_mem(lambda->param(0));
    const Enter* enter = cg.world().enter(lambda->param(0));
    cg.set_mem(enter->extract_mem());
    fun->frame_ = enter->extract_frame();

    for (size_t i = 0, e = fun->params().size(); i != e; ++i) {
        const Param* p = lambda->param(i+1);
        p->name = fun->param(i)->symbol().str();
        cg.emit(fun->param(i))->store(p);
    }

    cg.emit(fun->body(), continue_bb);
    cg.enter(continue_bb);
    cg.return1(lambda->params().back(), cg.get_mem());

    // go out of the lambda
    cg.cur_bb = next;
    cg.jump(exit_bb);
}

void BreakStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.break_target); }
void ContinueStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.jump(*cg.continue_target); }

void ReturnStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const {
    if (cg.is_reachable()) {
        const Param* ret_param = fun()->ret_param();
        if (const Call* call = expr()->isa<Call>()) {
            Array<const Def*> ops = cg.emit_ops(call);
            Array<const Def*> args(call->num_args() + 2);
            std::copy(ops.begin() + 1, ops.end(), args.begin() + 1);
            args[0] = cg.get_mem();
            args.back() = ret_param;
            cg.tail_call(ops[0], args);
        } else {
            if (expr()) 
                cg.return2(ret_param, cg.world().leave(cg.get_mem(), fun()->frame()), cg.emit(expr())->load());
            else
                cg.return1(ret_param, cg.world().leave(cg.get_mem(), fun()->frame()));
        }
    }
}

void ScopeStmt::emit(CodeGen& cg, JumpTarget& exit_bb) const { cg.emit(scope(), exit_bb); }

/*
 * items
 */

void FunItem::emit(CodeGen& cg) const { cg.emit_body(fun()); }
void TraitItem::emit(CodeGen& cg) const { assert( false && "todo"); }

} // namespace impala
