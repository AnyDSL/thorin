#ifndef ANYDSL_AIRNODE_H
#define ANYDSL_AIRNODE_H

#include <string>

#include "anydsl/air/enums.h"
#include "anydsl/util/cast.h"

namespace anydsl {

class AIRNode {
public:

    AIRNode(Nodekind nodekind, AIRNode* parent, const std::string& debug)
        : nodekind_(nodekind)
        , parent_(parent)
        , debug_(debug)
    {}
    virtual ~AIRNode() {}

    Nodekind nodekind() const { return nodekind_; }

    template<class T>
    T* getAs() { return dcast<T*>(this); }
    template<class T>
    const T* getAs() const { return dcast<T*>(this); }

    template<class T>
    T* parentAs() { return dcast<T>(parent_); }
    template<class T>
    const T* parentAs() const { return dcast<T>(parent_); }

    AIRNode* parent() { return parent_; }
    const AIRNode* parent() const { return parent_; }

#if 0
    template<class Child, class RetT>
    inline RetT accept(Visitor<Child,  RetT>* v);

    template<class Child, class RetT>
    inline RetT accept(ConstVisitor<Child,  RetT>* v) const;

    template<class Child, class RetT>
    inline RetT accept(DualConstVisitor<Child,  RetT>* v, const AIRNode* other) const;

#endif

private:

    Nodekind nodekind_;
    AIRNode* parent_;
    std::string debug_;

#if 0

#ifndef NDEBUG
public:
    void debug() const;
    void rdebug() const;
    void ptrdebug() const;
    virtual const char* debugType() const;
    virtual void debugInternals() const;
    virtual bool debugFast() const;
#else
public:
    void debug() const {}
    void rdebug() const {}
    void ptrdebug() const {}
#endif
#endif

};

} // namespace anydsl

#endif // ANYDSL_AIRNODE_H
