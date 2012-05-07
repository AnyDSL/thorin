#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include "anydsl/air/airnode.h"

namespace anydsl {

class PrimOp : public AIRNode {
protected:

    PrimOp(Nodekind nodekind, AIRNode* parent, const std::string& debug)
        : AIRNode(nodekind, parent, debug)
    {}
};


class ArithOp : public PrimOp {
public:

    ArithOp(AIRNode* parent, const std::string& debug)
        : PrimOp(Kind_ArithOp, parent, debug)
    {}

private:

    ArithOpKind kind_;
};

} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
