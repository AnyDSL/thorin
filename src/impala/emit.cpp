#include "impala/ast.h"

#include <boost/unordered_map.hpp>
#include <iostream>

#include "anydsl2/irbuilder.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/ref.h"
#include "anydsl2/type.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/world.h"

#include "impala/type.h"

using anydsl2::Array;
using anydsl2::ArrayRef;
using anydsl2::BB;
using anydsl2::Def;
using anydsl2::Ref;
using anydsl2::Symbol;
using anydsl2::Type;
using anydsl2::Var;
using anydsl2::World;
using anydsl2::make_name;

namespace impala {

class CodeGen {
public:

    CodeGen(World& world)
        : world(world)
        , root(new anydsl2::Fct(world))
        , curBB(root)
        , curFct(root)
    {}

    void fixto(BB* to) {
        if (reachable())
            curBB->fixto(to);
    }

    bool reachable() const { return curBB; }
    anydsl2::Fct* create_fct(const Lambda& lambda, Symbol symbol);

    World& world;
    anydsl2::AutoPtr<anydsl2::Fct> root;
    BB* curBB;
    anydsl2::Fct* curFct;
};

//------------------------------------------------------------------------------

void emit(World& world, const Prg* prg) {
    CodeGen cg(world);
    prg->emit(cg);
    cg.world.cleanup();
}

//------------------------------------------------------------------------------

anydsl2::Fct* CodeGen::create_fct(const Lambda& lambda, Symbol symbol) {
    size_t size = lambda.params().size();
    Array<Symbol> symbols(size);

    for_all2 (&sym, symbols, param, lambda.params())
        sym = param->symbol();

    size_t return_index = return_type(lambda.pi())->isa<NoRet>() ? size_t(-1) : lambda.pi()->size()-1;
    return lambda.air_fct_ = new anydsl2::Fct(world, lambda.pi(), symbols, return_index, symbol.str());
}

void Prg::emit(CodeGen& cg) const {
    for_all (f, fcts()) {
        anydsl2::Lambda* lambda = cg.create_fct(f->lambda(), f->symbol())->top();
        if (f->symbol() == Symbol("main"))
            lambda->set_extern();
            
        cg.root->nest(f->symbol(), f->lambda().air_fct());
    }

    for_all (f, fcts())
        f->emit(cg);
}

const anydsl2::Lambda* Lambda::emit(CodeGen& cg, BB* parent, const char* what) const {
    for_all (f, body()->fcts())
        air_fct()->nest(f->symbol(), cg.create_fct(f->lambda(), f->symbol()));

    air_fct()->set_parent(parent);
    BB* oldBB = cg.curBB;
    anydsl2::Fct* oldFct = cg.curFct;
    cg.curBB = cg.curFct = air_fct();

    body()->emit(cg);

    if (cg.reachable()) {
        if (is_continuation()) {
            std::cerr << what << " does not end with a call\n";
            cg.curBB->tail_call(cg.world.bottom(cg.world.pi0()), ArrayRef<const Def*>());
        } else {
            const Type* ret_type = return_type(pi());
            if (ret_type->isa<Void>())
                cg.curBB->return_void();
            else {
                std::cerr << what << " does not end with 'return'\n";
                cg.curBB->return_value(cg.world.bottom(ret_type));
            }
        }
    }

    cg.curFct->emit();

    cg.curBB  = oldBB;
    cg.curFct = oldFct;

    return air_fct()->top();
}

void Fct::emit(CodeGen& cg) const {
    lambda().emit(cg, cg.curBB, symbol().str());
    if (extern_)
        lambda().air_fct()->top()->set_extern();
}

Var* Decl::emit(CodeGen& cg) const {
    Var* var = cg.curBB->insert(symbol(), cg.world.bottom(type()));
    var->load()->debug = symbol().str();

    return var;
}

/*
 * Expr
 */

Array<const Def*> Expr::emit_ops(CodeGen& cg) const {
    Array<const Def*> defs(ops_.size());
    for_all2 (&def, defs, op, ops())
        def = op->emit(cg)->load();

    return defs;
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
    static int id = 0;
    cg.create_fct(lambda(), make_name("lambda", id++));
    return Ref::create(lambda().emit(cg, cg.curBB, "anonymous lambda expression"));
}

RefPtr Tuple::emit(CodeGen& cg) const {
    return Ref::create(cg.world.tuple(emit_ops(cg)));
}

RefPtr Id::emit(CodeGen& cg) const {
    return Ref::create(cg.curBB->lookup(symbol(), type()));
}

RefPtr PrefixExpr::emit(CodeGen& cg) const {
    switch (kind()) {
        case INC:
        case DEC: {
            RefPtr ref = rhs()->emit(cg);
            const Def* def = ref->load();
            const anydsl2::PrimType* pt = def->type()->as<anydsl2::PrimType>();
            const anydsl2::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
            const Def* ndef = cg.world.arithop(Token::to_arithop((TokenKind) kind()), def, one);
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
        const Def* rdef = rhs()->emit(cg)->load();

        // special case for 'a = expr' -> don't use lookup!
        RefPtr lref = op == Token::ASGN && id
                ? Ref::create(cg.curBB->insert(id->symbol(), cg.world.bottom(id->type())))
                : lhs()->emit(cg);

        const Def* ldef = lref->load();

        if (op != Token::ASGN) {
            TokenKind sop = Token::seperateAssign(op);
            rdef = cg.world.binop(Token::to_binop(sop), ldef, rdef);
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
    const anydsl2::PrimType* pt = def->type()->as<anydsl2::PrimType>();
    const anydsl2::PrimLit* one = cg.world.literal(pt->primtype_kind(), 1u);
    const Def* ndef = cg.world.arithop(Token::to_arithop((TokenKind) kind()), def, one);
    ref->store(ndef);

    return Ref::create(def);
}

RefPtr IndexExpr::emit(CodeGen& cg) const {
    return Ref::create(lhs()->emit(cg), index()->emit(cg)->load());
}

RefPtr Call::emit(CodeGen& cg) const {
    Array<const Def*> ops = emit_ops(cg);

    if (is_continuation_call()) {
        cg.curBB->tail_call(ops[0], ops.slice_back(1));
        cg.curBB = 0;
        return RefPtr(0);
    } else 
        return Ref::create(cg.curBB->call(ops[0], ops.slice_back(1), type()));
}

/*
 * Stmt
 */

void DeclStmt::emit(CodeGen& cg) const {
    if (cg.reachable()) {
        Var* var = decl()->emit(cg);
        if (const Expr* init_expr = init())
            var->store(init_expr->emit(cg)->load());
    }
}

void ExprStmt::emit(CodeGen& cg) const {
    if (cg.reachable())
        expr()->emit(cg);
}

void IfElseStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int cur_id = id++;

    // always create elseBB -- the edge from headBB to nextBB would be crtical anyway

    // create BBs
    BB* thenBB = cg.curFct->createBB(make_name("if-then", cur_id));
    BB* elseBB = cg.curFct->createBB(make_name("if-else", cur_id));

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
        BB* nextBB = cg.curFct->createBB(make_name("if-next", cur_id));
        thenCur->fixto(nextBB);
        elseCur->fixto(nextBB);
        nextBB->seal();
        cg.curBB = nextBB;
    }
}

void DoWhileStmt::emit(CodeGen& cg) const {
    static int id = 0;
    int cur_id = id++;

    // create BBs
    BB* bodyBB = cg.curFct->createBB(make_name("dowhile-body", cur_id));
    BB* condBB = cg.curFct->createBB(make_name("dowhile-cond", cur_id));
    BB* critBB = cg.curFct->createBB(make_name("dowhile-crit", cur_id));
    BB* nextBB = cg.curFct->createBB(make_name("dowhile-next", cur_id));

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
    int cur_id = id++;

    init()->emit(cg);

    // create BBs
    BB* headBB = cg.curFct->createBB(make_name("for-head", cur_id));
    BB* bodyBB = cg.curFct->createBB(make_name("for-body", cur_id));
    BB* stepBB = cg.curFct->createBB(make_name("for-step", cur_id));
    BB* nextBB = cg.curFct->createBB(make_name("for-next", cur_id));

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
    if (cg.reachable()) {
        if (const Call* call = expr()->isa<Call>()) {
            Array<const Def*> ops = call->emit_ops(cg);
            cg.curBB->return_tail_call(ops[0], ops.slice_back(1));
        } else {
            if (expr())
                cg.curBB->return_value(expr()->emit(cg)->load());
            else
                cg.curBB->return_void();
        }

        // all other statements in the same BB are unreachable
        cg.curBB = 0;
    }
}

void FctStmt::emit(CodeGen& cg) const {
    fct()->emit(cg);
}

void ScopeStmt::emit(CodeGen& cg) const {
    for_all (const &s, stmts())
        s->emit(cg);
}

} // namespace impala
