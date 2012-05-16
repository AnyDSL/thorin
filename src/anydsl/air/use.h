#ifndef ANYDSL_AIR_USE_H
#define ANYDSL_AIR_USE_H

#include "anydsl/air/airnode.h"

namespace anydsl {

class Def;

class Use : public AIRNode {
private:

    /// Do not create "default" \p Use instances.
    Use();
    /// Do not copy-construct a \p Use instance.
    Use(const Use&);
    /// Do not copy-assign a \p Use instance.
    Use& operator = (const Use&);

public:

    Use(Def* def, AIRNode* parent, const std::string& debug = "");
    virtual ~Use();

    const Def* def() const { return def_; }

    AIRNode* parent() { return parent_; }
    const AIRNode* parent() const { return parent_; }
    template<class T> T* parentAs() { return dcast<T>(parent_); }
    template<class T> const T* parentAs() const { return dcast<T>(parent_); }

private:

    Def* def_;
    AIRNode* parent_;
};

} // namespace anydsl

#endif // ANYDSL_AIR_USE_H
