#ifndef IMPALA_EMITTER_H
#define IMPALA_EMITTER_H

#include <vector>

#include <impala/environment.h>
#include <impala/value.h>

namespace anydsl {
    union Box;
    class CExpr;
    class Def;
    class Fix;
    class Lambda;
    class Param;
    class Position;
    class Symbol;
    class Type;
}

namespace impala {

class BB;
class Fct;
class Token;

struct Cursor {

    Cursor() {}
    Cursor(Fct* fct, BB* bb) 
        : fct(fct)
        , bb(bb)
    {}

    Fct* fct;
    BB* bb;
};

class Emitter {
public:

    // prologue and epilogue
    void prologue(const anydsl::Position& pos);
    anydsl::Lambda* exit();

    // helpers
    Value appendLambda(anydsl::CExpr* cexpr, anydsl::Type* type);
    Value decl(const Token& tok, anydsl::Type* type);
    Value param(const Token& tok, anydsl::Type* type, anydsl::Param* p);
    Fct* fct(Cursor& old, const anydsl::Location& loc, const Token& name);
    void fixBB(const anydsl::Location& loc, BB* newBB);
    void glueTo(const anydsl::Location& loc, BB* to);
    void returnStmt(const anydsl::Location& loc, Value retVal);

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

    Cursor getCursor() { return cursor_; }
    void setCursor(Cursor cursor) { cursor_ = cursor; }
    void setCursor(BB* bb) { cursor_.bb = bb; }

    BB* bb() { return cursor_.bb; }
    Fct* fct() { return cursor_.fct; }

private:

    Fct* root_;
    Cursor cursor_;
    Fct* main_;      ///< main function; 0 until not found
    Environment env_;///< keep track of symbols and its associated anydsl::Type
};

} // namespace impala

#endif // IMPALA_EMITTER_H
