#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <string>

#include "anydsl/air/enums.h"
#include "anydsl/util/cast.h"

namespace anydsl {

class AIRNode {
protected:

    AIRNode(IndexKind index, const std::string& debug)
        : index_(index)
        , debug_(debug)
    {}
    virtual ~AIRNode() {}

public:

    IndexKind index() const { return index_; }
    NodeKind nodeKind() const { return (NodeKind) index_; }

    template<class T> T* getAs() { return dcast<T*>(this); }
    template<class T> const T* getAs() const { return dcast<T*>(this); }

    std::string debug() const { return debug_; }

#if 0
    template<class Child, class RetT>
    inline RetT accept(Visitor<Child,  RetT>* v);

    template<class Child, class RetT>
    inline RetT accept(ConstVisitor<Child,  RetT>* v) const;

    template<class Child, class RetT>
    inline RetT accept(DualConstVisitor<Child,  RetT>* v, const AIRNode* other) const;
#endif

private:

    IndexKind index_;

protected:

    std::string debug_;
};

} // namespace anydsl

#endif // ANYDSL_AIRNODE_H
