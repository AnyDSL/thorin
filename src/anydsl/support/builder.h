#ifndef ANYDSL_SUPPORT_BUILDER_H
#define ANYDSL_SUPPORT_BUILDER_H

#include "anydsl/air/primop.h"

namespace anydsl {

class Universe;

class ArithOp;

class Builder {
public:

    Builder(Universe& universe);
    ~Builder();


    ArithOp* createArithOp(ArithOpKind arithOpKind,
                           Def* ldef, Def* rdef, 
                           const std::string& ldebug = "", 
                           const std::string& rdebug = "", 
                           const std::string&  debug = "");

    const Universe& universe() const { return universe_; } 

private:

    Universe& universe_;
};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_BUILDER_H
