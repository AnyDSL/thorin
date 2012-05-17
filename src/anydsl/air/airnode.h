#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <string>

#include "anydsl/air/enums.h"
#include "anydsl/util/cast.h"

namespace anydsl {

class AIRNode {
private:
    AIRNode() {}
    friend class Use;

protected:

    AIRNode(IndexKind index)
        : index_(index)
    {}
    virtual ~AIRNode() {}

public:

    IndexKind index() const { return index_; }
    NodeKind nodeKind() const { return (NodeKind) index_; }

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

    mutable std::string debug;

private:

    IndexKind index_;

protected:
};

} // namespace anydsl

#endif // ANYDSL_AIRNODE_H
