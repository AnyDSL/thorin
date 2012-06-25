#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <string>

#include "anydsl/enums.h"
#include "anydsl/util/cast.h"

namespace anydsl {

class AIRNode {
protected:

    AIRNode(IndexKind indexKind)
        : indexKind_(indexKind)
    {}
    virtual ~AIRNode() {}

public:

    IndexKind indexKind() const { return indexKind_; }
    void dump() const;

    ANYDSL_MIXIN_AS_ISA

    /**
     * Just do what ever you want with this field.
     * Perhaps you want to attach file/line/col information in this field.
     * \p Location provides convenient functionality to achieve this.
     */
    mutable std::string debug;

private:

    IndexKind indexKind_;
};

} // namespace anydsl

#endif
