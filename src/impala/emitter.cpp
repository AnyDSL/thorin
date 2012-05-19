#include <impala/emitter.h>

#include "anydsl/air/lambda.h"
#include "anydsl/air/literal.h"
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


void Emitter::returnStmt(Value retVal) {
    fct()->insertReturn(cursor.bb, retVal.load());
}

Value Emitter::decl(const Token& tok, const Type* type) {
    Symbol sym = tok.symbol();

    if (/*Type* prev =*/ env_.clash(sym)) {
        tok.error() << "symbol '" << sym.str() << "' already defined in this scope\n";
        /*prev->error()*/ std::cerr << "previous definition here\n";
        anydsl_assert(bb()->hasVN(sym), "env and value map out of sync");
        Binding* bind = bb()->getVN(sym, type, false);

        return Value(bind);
    }

    Binding* bind = new Binding(sym, world_.undef(type));
    bb()->setVN(bind);
    env_.insert(sym, type);

    return Value(bind);
}

Value Emitter::param(const Token& tok, const Type* type, Param* p) {
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

    return Value(bind);
}

Value Emitter::literal(const Token& tok) {
    switch (tok) {
        case Token::TRUE:  return Value(world_.literal(true));
        case Token::FALSE: return Value(world_.literal(false));

        default:
            switch (tok) {
#define IMPALA_LIT(TOK, T) \
                case Token:: TOK: return Value(world_.literal(type2kind<T>::kind, tok.box()));
#include <impala/tokenlist.h>
                default: ANYDSL_UNREACHABLE;
            }
    }
}

Value Emitter::prefixOp(const Token& op, Value bval) {
    if (op == Token::ADD)
        return bval; // this is a NOP

    if (const PrimType* p = bval.type()->isa<PrimType>()) {
        switch (op) {
            case Token::SUB: {
                // TODO incorrect for f32, f64
                PrimLit* zero = world_.literal(p->kind(), 0u);
                return Value(world_.createArithOp(anydsl::ArithOp_sub, zero, bval.load()));
            }
            case Token::INC:
            case Token::DEC: {
                PrimLit* one = world_.literal(p->kind(), 1u);
                Value val(world_.createArithOp(op.toArithOp(), bval.load(), one));
                bval.store(val.load());
                return bval;
            }
            default: ANYDSL_UNREACHABLE; // TODO
        }
    }

    return Value(world_.error());
}

Value Emitter::infixOp(Value aval, const Token& op, Value bval) {
    const PrimType* p1 = aval.type()->isa<PrimType>();
    const PrimType* p2 = bval.type()->isa<PrimType>();

    if (p1 && p1 == p2) {
        if (op.isAsgn()) {
            Value val = bval;

            if (op != Token::ASGN) {
                Token tok = op.seperateAssign();
                val = infixOp(aval, tok, bval);
            }

            aval.store(val.load());
            return aval;
        }

        if (op.isArith())
            return Value(world_.createArithOp(op.toArithOp(), aval.load(), bval.load()));
        else if (op.isRel())
            return Value(world_.createRelOp(op.toRelOp(), aval.load(), bval.load()));
    }

    op.error() << "type error: TODO\n";

    const Type* t = p1 ? p1 : p2;
    if (t)
        return Value(world_.literal_error(t)); 

    return Value(world_.error());
}

#if 0

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

#endif

} // namespace impala

