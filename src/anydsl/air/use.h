#ifndef ANYDSL_USE_H
#define ANYDSL_USE_H

#include "anydsl/air/airnode.h"

namespace anydsl {

class Def;

class Use : public AIRNode {
private:

    Use();
    Use(const Use&);
    Use& operator = (const Use&);

public:

    Use(Def* def, AIRNode* parent, const std::string& debug = "")
        : AIRNode(Kind_Use, parent, debug)
        , def_(0) 
    {
        setDef(def);
    }

    ~Use() { unsetDef(); }

    /*
     * getters & setters
     */

    Def* def() const { return def_; }
    // does not return a reference - we don't want to allow direct modification
    // roland: well it is a reference, correct?
    Def* def() { return def_; }

    void set(Def* def) { unsetDef(); setDef(def); }
    void set(const Use& use) { unsetDef(); setDef(use.def()); }
    bool isSet() const { return def_ != 0; }

    //void operator=(Def* def) { set(def); }
    //void operator=(const Use &use) { set(use); }

#if 0
    inline Type* type();
    inline const Type* type() const;
#endif

private:

    void setDef(Def* def);
    void unsetDef();

    Def* def_;

    //Def is incomplete - cannot use this:
    //    friend Use* Def::useMe(const Location&);
    friend class Def;

#if 0
    ANYDSL_DEBUG_FUNCTIONS;
#endif
};

} // namespace anydsl

#endif // ANYDSL_USE_H
