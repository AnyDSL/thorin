#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <cassert>
#include <string>

#include "anydsl/enums.h"

namespace anydsl {

class Printer;

class AIRNode : public MagicCast {
protected:

    AIRNode(int kind)
        : kind_(kind)
    {}
    AIRNode(IndexKind indexKind)
        : kind_(indexKind)
    {}

public:

    int kind() const { return kind_; }
    bool isCoreNode() const { return ::anydsl::isCoreNode(kind()); }
    bool isPrimType() const { return ::anydsl::isPrimType(kind()); }
    bool isArithOp()  const { return ::anydsl::isArithOp(kind()); }
    bool isRelOp()    const { return ::anydsl::isRelOp(kind()); }
    bool isConvOp()   const { return ::anydsl::isConvOp(kind()); }
    bool isType() const;

    IndexKind indexKind() const { assert(isCoreNode()); return (IndexKind) kind_; }

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    /**
     * Just do what ever you want with this field.
     * Perhaps you want to attach file/line/col information in this field.
     * \p Location provides convenient functionality to achieve this.
     */
    mutable std::string debug;

private:

    int kind_;
};

} // namespace anydsl

#endif
