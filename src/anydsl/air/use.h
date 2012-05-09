#ifndef ANYDSL_AIR_USE_H
#define ANYDSL_AIR_USE_H

#include "anydsl/air/airnode.h"

namespace anydsl {

class Def;

class Use : public AIRNode {
private:

    Use();
    Use(const Use&);
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

    //friend class Def;

#if 0
    ANYDSL_DEBUG_FUNCTIONS;
#endif
};

} // namespace anydsl

#endif // ANYDSL_AIR_USE_H
