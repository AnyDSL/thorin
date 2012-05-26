#ifndef ANYDSL_AIR_AIRNODE_H
#define ANYDSL_AIR_AIRNODE_H

#include <string>

#include "anydsl/enums.h"
#include "anydsl/util/cast.h"

namespace anydsl {

class AIRNode {
protected:

    AIRNode(IndexKind index)
        : index_(index)
    {}
    virtual ~AIRNode() {}

public:

    IndexKind index() const { return index_; }

    template<class T> T* as()  { return scast<T>(this); }
    template<class T> T* isa() { return dcast<T>(this); }
    template<class T> const T* as()  const { return scast<T>(this); }
    template<class T> const T* isa() const { return dcast<T>(this); }

#if 0
    template<class Child, class RetT>
    inline RetT accept(Visitor<Child,  RetT>* v);

    template<class Child, class RetT>
    inline RetT accept(ConstVisitor<Child,  RetT>* v) const;

    template<class Child, class RetT>
    inline RetT accept(DualConstVisitor<Child,  RetT>* v, const AIRNode* other) const;
#endif

    /**
     * Just do what ever you want with this field.
     * Perhaps you want to attach file/line/col information in this field.
     * \p Location provides convenient functionality to achieve this.
     */
    mutable std::string debug;

private:

    IndexKind index_;
};

} // namespace anydsl

#endif // ANYDSL_AIR_AIRNODE_H
