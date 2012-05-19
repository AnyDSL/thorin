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

struct Cursor {

    Cursor() {}
    Cursor(anydsl::Fct* fct, anydsl::BB* bb) 
        : fct(fct)
        , bb(bb)
    {}

    anydsl::Fct* fct;
    anydsl::BB* bb;
};

class Emitter {
public:

    Emitter(anydsl::World& world) 
        : world_(world) 
    {}

    // prologue and epilogue
    void prologue();
    anydsl::Lambda* exit();

    // helpers
    Value decl(const Token& tok, const anydsl::Type* type);
    Value param(const Token& tok, const anydsl::Type* type, anydsl::Param* p);
    anydsl::Fct* fct(Cursor& old, const anydsl::Pi* pi, const Token& name);
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

    anydsl::BB* bb() { return cursor.bb; }
    anydsl::Fct* fct() { return cursor.fct; }

    Cursor cursor;

private:

    anydsl::World& world_;
    anydsl::Fct* root_;
    anydsl::Fct* main_; ///< main function; 0 until not found
    Environment env_;   ///< keep track of symbols and its associated anydsl::Type
};

} // namespace impala

#endif // IMPALA_EMITTER_H
