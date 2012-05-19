#include <impala/emitter.h>

#include "anydsl/air/lambda.h"
#include "anydsl/air/world.h"
#include "anydsl/util/location.h"
#include "anydsl/support/cfg.h"
#include "anydsl/support/binding.h"

#include "impala/token.h"

using namespace anydsl;

namespace impala {

void Emitter::prologue() {
    root_ = new Fct(0, world_.pi(), Symbol("<root-function>"));
    cursor = Cursor(root_, root_);
    main_ = 0;
}

Lambda* Emitter::exit() {
    if (main_)
        root_->goesto(main_); // TODO exit
    else
        root_->invokes(root_->lambda()); // TODO exit

    return root_->lambda_;
}

Fct* Emitter::fct(Cursor& old, const Pi* pi, const Token& name) {
    old = cursor;
    cursor.bb = cursor.fct = root_->createSubFct(pi, name.symbol());

    if (std::string("main") == name.symbol().str()) {
        if (main_)
            name.error() << "main already defined\n";
        else
            main_ = fct();
    }

    return fct();
}

}
#if 0


void Emitter::returnStmt(Value retVal) {
    fct()->insertReturn(cursor_.bb, retVal.load());
}

//------------------------------------------------------------------------------

Value Emitter::decl(const Token& tok, Type* type) {
    Location loc(tok.pos1(), type->pos2());

    Symbol sym = tok.symbol();

    if (Type* prev = env_.clash(sym)) {
        tok.error() << "symbol '" << sym.str() << "' already defined in this scope\n";
        prev->error() << "previous definition here\n";
        anydsl_assert(bb()->hasVN(sym), "env and value map out of sync");
        Binding* bind = bb()->getVN(sym, type, false);

        return Value(bind);
    }

    Undef* undef = new Undef(loc);
    undef->meta.set(type);
    Binding* bind = new Binding(sym, undef);
    bb()->setVN(bind);
    env_.insert(sym, type);

    return Value(bind);
}

Value Emitter::param(const Token& tok, Type* type, Param* p) {
    Location loc(tok.pos1(), type->pos2());
    Symbol sym = tok.symbol();

#if 0
    if (Binding* prev = env_.clash(sym)) {
        tok.error() << "symbol '" << sym.str() << "' already defined in this scope\n";
        prev->error() << "previous definition here\n";
        return prev;
    }
#endif

    Binding* bind = new Binding(sym, p);
    bb()->setVN(bind);
    env_.insert(sym, type);
    //cursor_.bb->values_[sym] = p;

    return Value(bind);
}

Value Emitter::literal(const Token& tok) {
    Primitive* lit = new Primitive(tok.loc());

    switch (tok) {
        case Token::TRUE:
        case Token::FALSE:
            lit->box().bool_ = tok == Token::TRUE;
            lit->setType(PrimitiveType::Type_boolean);
            return Value(lit);

        default:
            lit->box() = tok.box();
            switch (tok) {
#define IMPALA_LIT(tok, t) \
                case Token:: tok: \
                    lit->setType(PrimitiveType::Type_ ## t); \
                    return Value(lit);
#include <impala/tokenlist.h>

                default: ANYDSL_UNREACHABLE;
            }
    }
}

Value Emitter::prefixOp(const Token& op, Value bval) {
    if (op == Token::ADD)
        return bval; // this is a NOP

    Location loc(op.pos1(), bval.pos2());

    if (const PrimitiveType* t = dcast<PrimitiveType>(bval.type())) {
        Beta* beta = new Beta(loc); 

    // TODO incorrect for f32, f64
#define CASE_PRE_SUB_TYPE(type) \
    case PrimitiveType:: Type_ ## type: { \
        Primitive* zero = Primitive::create<PrimitiveType::Type_ ## type >(0); \
        beta->args().push_back(zero); \
        beta->args().push_back(bval.load()); \
        Def* fct = new Intrinsic(op.loc(), Intrinsic::BIN_SUB_ ## type ## _ ## type ); \
        Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
        beta->fct.set(fct); \
        return appendLambda(beta, retT); \
    }

#define CASE_PRE_INC_TYPE(type) \
    case PrimitiveType:: Type_ ## type: { \
        Primitive* one = Primitive::create<PrimitiveType::Type_ ## type >(1); \
        beta->args().push_back(bval.load()); \
        beta->args().push_back(one); \
        Def* fct = new Intrinsic(op.loc(), Intrinsic::BIN_ADD_ ## type ## _ ## type ); \
        Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
        beta->fct.set(fct); \
        Value val = appendLambda(beta, retT); \
        bval.store(val.load()); \
        return bval; \
    }

#define CASE_PRE_DEC_TYPE(type) \
    case PrimitiveType:: Type_ ## type: { \
        Primitive* one = Primitive::create<PrimitiveType::Type_ ## type >(1); \
        beta->args().push_back(bval.load()); \
        beta->args().push_back(one); \
        Def* fct = new Intrinsic(op.loc(), Intrinsic::BIN_SUB_ ## type ## _ ## type ); \
        Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
        beta->fct.set(fct); \
        Value val = appendLambda(beta, retT); \
        bval.store(val.load()); \
        return bval; \
    }

#define CASE_PRE_OP(op) \
    case Token:: op : \
        switch (t->which()) {             \
            CASE_PRE_ ## op ## _TYPE(i8)  \
            CASE_PRE_ ## op ## _TYPE(i16) \
            CASE_PRE_ ## op ## _TYPE(i32) \
            CASE_PRE_ ## op ## _TYPE(i64) \
            CASE_PRE_ ## op ## _TYPE(u8)  \
            CASE_PRE_ ## op ## _TYPE(u16) \
            CASE_PRE_ ## op ## _TYPE(u32) \
            CASE_PRE_ ## op ## _TYPE(u64) \
            CASE_PRE_ ## op ## _TYPE(f32) \
            CASE_PRE_ ## op ## _TYPE(f64) \
            default: ANYDSL_UNREACHABLE;  \
        }

        switch (op) {
            CASE_PRE_OP(SUB)
            //CASE_PRE_OP(MUL) // TODO: deref
            //CASE_PRE_OP(AND) // TODO: address of
            //CASE_PRE_OP(NOT) // TODO: bit not
            //CASE_PRE_OP(L_N) // TODO: logic not
            CASE_PRE_OP(INC)
            CASE_PRE_OP(DEC)
            default: ANYDSL_UNREACHABLE;
        }
    }

    op.error() << "type error: TODO\n";

    return Value(new ErrorValue(loc));
}

Value Emitter::infixOp(Value aval, const Token& op, Value bval) {
    // TODO type checking
    if (op.isAsgn()) {
        Value val = bval;

        if (op != Token::ASGN) {
            Token tok = op.seperateAssign();
            val = infixOp(aval, tok, bval);
        }

        aval.store(val.load());
        return aval;
    }

    Type* t1 = aval.type();
    Type* t2 = bval.type();

    Location loc(Location(aval.pos1(), bval.pos2()));
    Beta* beta  = new Beta(loc);
    UseList& args = beta->args();
    args.push_back(aval.load());
    args.push_back(bval.load());

    const PrimitiveType* p1 = dcast<PrimitiveType>(t1);
    const PrimitiveType* p2 = dcast<PrimitiveType>(t2);

    if (p1 && p2 && p1->which() == p2->which()) {

#define CASE_INFIX_BIN(op, type) \
        case PrimitiveType:: Type_ ## type : { \
            Intrinsic::Which intrin = Intrinsic::FUN_BIN_ ## op ; \
            Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
            beta->fct.set(new Intrinsic(intrin)); \
            return appendLambda(beta, retT); \
        }


#define CASE_INFIX_REL(op, type) \
        case PrimitiveType:: Type_ ## type : { \
            Intrinsic::Which intrin = Intrinsic::FUN_BIN_ ## op ; \
            Type* retT = new PrimitiveType(PrimitiveType::Type_boolean); \
            beta->fct.set(new Intrinsic(intrin)); \
            return appendLambda(beta, retT); \
        }


#define CASE_INFIX_OP(kind, op) \
        case Token:: op : \
            switch (p1->which()) {           \
                CASE_INFIX_ ## kind(op, i8)  \
                CASE_INFIX_ ## kind(op, i16) \
                CASE_INFIX_ ## kind(op, i32) \
                CASE_INFIX_ ## kind(op, i64) \
                CASE_INFIX_ ## kind(op, u8)  \
                CASE_INFIX_ ## kind(op, u16) \
                CASE_INFIX_ ## kind(op, u32) \
                CASE_INFIX_ ## kind(op, u64) \
                CASE_INFIX_ ## kind(op, f32) \
                CASE_INFIX_ ## kind(op, f64) \
                default: ANYDSL_UNREACHABLE; \
            }

#define CASE_INFIX_INT_OP(kind, op) \
        case Token:: op : \
            switch (p1->which()) {           \
                CASE_INFIX_ ## kind(op, i8)  \
                CASE_INFIX_ ## kind(op, i16) \
                CASE_INFIX_ ## kind(op, i32) \
                CASE_INFIX_ ## kind(op, i64) \
                CASE_INFIX_ ## kind(op, u8)  \
                CASE_INFIX_ ## kind(op, u16) \
                CASE_INFIX_ ## kind(op, u32) \
                CASE_INFIX_ ## kind(op, u64) \
                default: ANYDSL_UNREACHABLE; \
            }

        switch (op) {
            CASE_INFIX_OP(BIN, ADD)
            CASE_INFIX_OP(BIN, SUB)
            CASE_INFIX_OP(BIN, MUL)
            CASE_INFIX_OP(BIN, DIV)
            CASE_INFIX_OP(BIN, MOD)
            CASE_INFIX_OP(REL,  EQ)
            CASE_INFIX_OP(REL, NEQ)
            CASE_INFIX_OP(REL,  LT)
            CASE_INFIX_OP(REL, LEQ)
            CASE_INFIX_OP(REL,  GT)
            CASE_INFIX_OP(REL, GEQ)
            CASE_INFIX_INT_OP(BIN, AND)
            CASE_INFIX_INT_OP(BIN,  OR)
            CASE_INFIX_INT_OP(BIN, XOR)
            CASE_INFIX_INT_OP(BIN, SHL)
            CASE_INFIX_INT_OP(BIN, SHR)
            default: ANYDSL_UNREACHABLE;
        }
    }

    op.error() << "type error: TODO\n";

    return Value(new ErrorValue(loc));
}

Value Emitter::postfixOp(Value aval, const Token& op) {
    Location loc(aval.pos1(), op.pos2());

    if (const PrimitiveType* t = dcast<PrimitiveType>(aval.type())) {
        Beta* beta = new Beta(loc); 

#define CASE_POST_INC_TYPE(type) \
    case PrimitiveType:: Type_ ## type: { \
        Primitive* one = Primitive::create<PrimitiveType::Type_ ## type >(1); \
        beta->args().push_back(aval.load()); \
        beta->args().push_back(one); \
        Def* fct = new Intrinsic(op.loc(), Intrinsic::BIN_ADD_ ## type ## _ ## type ); \
        Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
        beta->fct.set(fct); \
        Value val = appendLambda(beta, retT); \
        aval.store(val.load()); \
        return val; \
    }

#define CASE_POST_DEC_TYPE(type) \
    case PrimitiveType:: Type_ ## type: { \
        Primitive* one = Primitive::create<PrimitiveType::Type_ ## type >(1); \
        beta->args().push_back(aval.load()); \
        beta->args().push_back(one); \
        Def* fct = new Intrinsic(op.loc(), Intrinsic::BIN_SUB_ ## type ## _ ## type ); \
        Type* retT = new PrimitiveType(PrimitiveType::Type_ ## type ); \
        beta->fct.set(fct); \
        Value val = appendLambda(beta, retT); \
        aval.store(val.load()); \
        return val; \
    }

#define CASE_POST_OP(op) \
    case Token:: op : \
        switch (t->which()) {              \
            CASE_POST_ ## op ## _TYPE(i8)  \
            CASE_POST_ ## op ## _TYPE(i16) \
            CASE_POST_ ## op ## _TYPE(i32) \
            CASE_POST_ ## op ## _TYPE(i64) \
            CASE_POST_ ## op ## _TYPE(u8)  \
            CASE_POST_ ## op ## _TYPE(u16) \
            CASE_POST_ ## op ## _TYPE(u32) \
            CASE_POST_ ## op ## _TYPE(u64) \
            CASE_POST_ ## op ## _TYPE(f32) \
            CASE_POST_ ## op ## _TYPE(f64) \
            default: ANYDSL_UNREACHABLE;   \
        }

        switch (op) {
            CASE_POST_OP(INC)
            CASE_POST_OP(DEC)
            default: ANYDSL_UNREACHABLE;
        }
    }

    op.error() << "type error: TODO\n";

    return Value(new ErrorValue(loc));
}

Value Emitter::fctCall(Value f, std::vector<Value> args) {
    Beta* beta = new Beta(Location(f.pos1(), args.back().pos2()));
    beta->fct.set(f.load());

    FOREACH(arg, args)
        beta->args().push_back(arg.load());

    return appendLambda(beta, 0 /*TODO*/);
}

Type* Emitter::builtinType(const Token& tok) {
    switch (tok) {
#define IMPALA_TYPE(itype, atype) \
        case Token:: TYPE_ ## itype: \
            return new PrimitiveType(tok.loc(), PrimitiveType:: Type_ ## atype);
#include <impala/tokenlist.h>

        default: ANYDSL_UNREACHABLE;
    }
}

Value Emitter::id(const Token& tok) {
    const Symbol sym = tok.symbol();

    Type* type = env_.lookup(sym);

    if (!type) {
        anydsl_assert(!bb()->hasVN(sym), "env and value map out of sync");

        tok.error() << "symbol '" << sym.str() << "' not defined in current scope\n";
        ErrorValue* error = new ErrorValue(tok.loc());
        error->meta.set(type);

        return Value(error);
    }

    Binding* bind = bb()->getVN(tok.loc(), tok.symbol(), type, false);

    if (bind)
        return Value(bind);

    tok.error() << "symbol '" << sym << "' not defined in current scope\n";
    return Value(new Undef(tok.loc()));
}

} // namespace impala

#endif
