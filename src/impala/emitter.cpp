#include <impala/emitter.h>

#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/world.h"
#include "anydsl/cfg.h"
#include "anydsl/binding.h"
#include "anydsl/util/location.h"

#include "impala/token.h"

using namespace anydsl;

namespace impala {

Emitter::Emitter(anydsl::World& world) 
    : world(world) 
    , main_(0)
{
    root_ = world.createLambda();
}

Lambda* Emitter::exit() {
#if 0
    if (main_)
        root_->goesto(main_); // TODO exit
    else
        root_->invokes(root_->lambda()); // TODO exit
#endif

    return root_;
}

Fct* Emitter::fct(const Token& name) {
    curBB = curFct = new Fct(world, name.symbol());

    if (std::string("main") == name.symbol().str()) {
        if (main_)
            name.error() << "main already defined\n";
        else
            main_ = curFct;
    }

    return curFct;
}


void Emitter::returnStmt(Value retVal) {
    curFct->insertReturnStmt(curBB, retVal.load());
}

Value Emitter::decl(const Token& tok, const Type* type) {
    Symbol sym = tok.symbol();

    if (/*Type* prev =*/ env_.clash(sym)) {
        tok.error() << "symbol '" << sym.str() << "' already defined in this scope\n";
        /*prev->error()*/ std::cerr << "previous definition here\n";
        anydsl_assert(curBB->hasVN(sym), "env and value map out of sync");
        Binding* bind = curBB->getVN(sym, type, false);

        return Value(bind);
    }

    Binding* bind = new Binding(sym, world.undef(type));
    curBB->setVN(bind);
    env_.insert(sym, type);
    return Value(bind);
}

void Emitter::param(const Token& tok, const Type* type) {
    const Symbol name = tok.symbol();

    if (/*const Type* prev =*/ env_.clash(name)) {
        tok.error() << "symbol '" << name.str() << "' already defined in this scope\n";
        /*prev->error()*/ std::cerr << "previous definition here\n";
        return;
    }
#if 0

    Param* p = *curFct->lambda()->appendParam(type);
    p->debug = name.str();
    Binding* bind = new Binding(name, p);
    curBB->setVN(bind);
    env_.insert(name, type);
#endif
}

void Emitter::fixto(anydsl::BB* to) {
    curBB->fixto(to);
    curBB = to;
}

Value Emitter::literal(const Token& tok) {
    switch (tok) {
        case Token::TRUE:  return Value(world.literal(true));
        case Token::FALSE: return Value(world.literal(false));

        default:
            switch (tok) {
#define IMPALA_LIT(TOK, T) \
                case Token:: TOK: return Value(world.literal_##T(tok.box()));
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
                const PrimLit* zero = world.literal(p->kind(), 0u);
                return Value(world.createArithOp(anydsl::ArithOp_sub, zero, bval.load()));
            }
            case Token::INC:
            case Token::DEC: {
                const PrimLit* one = world.literal(p->kind(), 1u);
                Value val(world.createArithOp(op.toArithOp(), bval.load(), one));
                bval.store(val.load());
                return bval;
            }
            default: ANYDSL_UNREACHABLE; // TODO
        }
    }

    return error();
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
            return Value(world.createArithOp(op.toArithOp(), aval.load(), bval.load()));
        else if (op.isRel())
            return Value(world.createRelOp(op.toRelOp(), aval.load(), bval.load()));
    }

    op.error() << "type error: TODO\n";

    const Type* t = p1 ? p1 : p2;
    if (t)
        return Value(world.literal_error(t)); 

    return error();
}

Value Emitter::postfixOp(Value aval, const Token& op) {
    if (const PrimType* p = aval.type()->isa<PrimType>()) {
        const PrimLit* one = world.literal(p->kind(), 1u);
        Value val(world.createArithOp(op.toArithOp(), aval.load(), one));
        aval.store(val.load());
        return val;
    }

    op.error() << "type error: TODO\n";

    return error();
}

Value Emitter::fctCall(Value f, std::vector<Value> args) {
#if 0
    Beta* beta = new Beta(Location(f.pos1(), args.back().pos2()));
    beta->fct.set(f.load());

    FOREACH(arg, args)
        beta->args().push_back(arg.load());
#endif

    return error();
}

const Type* Emitter::builtinType(const Token& tok) {
    return world.type(tok.toPrimType());
}

Value Emitter::id(const Token& tok) {
    const Symbol sym = tok.symbol();

    const Type* type = env_.lookup(sym);

    if (!type) {
        anydsl_assert(!curBB->hasVN(sym), "env and value map out of sync");

        tok.error() << "symbol '" << sym.str() << "' not defined in current scope\n";
        return error();
    }

    Binding* bind = curBB->getVN(tok.symbol(), type, false);

    if (bind)
        return Value(bind);

    tok.error() << "symbol '" << sym << "' not defined in current scope\n";

    return Value(world.undef(type));
}

Value Emitter::error() {
    return Value(world.literal_error(world.type_u32()));
}

} // namespace impala

