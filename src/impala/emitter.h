#ifndef IMPALA_EMITTER_H
#define IMPALA_EMITTER_H

#include <vector>

#include "impala/environment.h"
#include "impala/value.h"

namespace anydsl {
    class BB;
    class Fct;
    class Lambda;
    class Param;
    class Pi;
    class Type;
    class World;
    union Box;
}

namespace impala {

class Token;

class Emitter {
public:

    Emitter(anydsl::World& world);

    void prologue();
    anydsl::Lambda* exit();

    // helpers
    Value decl(const Token& tok, const anydsl::Type* type);
    void param(const Token& tok, const anydsl::Type* type);
    anydsl::Fct* fct(const anydsl::Pi* pi, const Token& name);
    void glueTo(anydsl::BB* to);
    void returnStmt(Value retVal);

    // expressions
    Value literal(const Token& tok);
    Value prefixOp(const Token& op, Value bval);
    Value infixOp(Value aval, const Token& op, Value bval);
    Value postfixOp(Value aval, const Token& op); 
    Value fctCall(Value f, std::vector<Value> args);
    anydsl::Type* builtinType(const Token& tok);
    Value id(const Token& tok);

    void pushScope() { env_.pushScope();  }
    void popScope()  { env_.popScope();  }

    anydsl::Fct* curFct;
    anydsl::BB* curBB;

    Value error();

private:

    anydsl::World& world_;
    anydsl::Lambda* root_;
    anydsl::Fct* main_; ///< main function; 0 until not found
    Environment env_;   ///< keep track of symbols and its associated anydsl::Type
};

} // namespace impala

#endif // IMPALA_EMITTER_H
