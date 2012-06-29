#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <string>

#include "anydsl/enums.h"

namespace anydsl {

class Printer;

class AIRNode : public MagicCast {
protected:

    AIRNode(IndexKind indexKind)
        : indexKind_(indexKind)
    {}

public:

    IndexKind indexKind() const { return indexKind_; }

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

    IndexKind indexKind_;
};

} // namespace anydsl

#endif
