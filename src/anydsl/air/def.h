#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <string>
#include <vector>

#include "anydsl/air/airnode.h"

namespace anydsl {

class Use;

class Def : public AIRNode {
private:

    ~Def() { anydsl_warn_assert(false, "TODO: Detach all uses in Def destructor"); }

public:

    typedef std::vector<Use*> Uses;

    Def(Nodekind nodekind, AIRNode* parent, const std::string& debug)
        : AIRNode(nodekind, parent, debug) 
    {}

    /// Creates a new \p Use object pointing to \p this definition
    Use* useMe();

    /**
     * Manually adds given \p Use object to the list of uses of this \p Def.
     * use->def() must already point to \p this .
     */
    void registerUse(Use* use);

    /**
     * Manually removes given \p Use object from the list of uses of this \p Def.
     * use->def() must point to \p this , but should be unset right after the call to this function
     */
    void unregisterUse(Use* use);

    /*
        TODO: What if noone is using given Def anymore. delete?
        Or maybe put on a list "todelete" (and remove if some use appears)
        Otherwise bad things can happen when we try to swap uses
        */

    bool lt(const Def* /*other*/) const {
        ANYDSL_NOT_IMPLEMENTED;
        return false;
    }

    const Uses& uses() const { return uses_; }

private:
    Uses uses_;

#if 0
    ANYDSL_DEBUG_FUNCTIONS;
#endif
};

} // namespace anydsl

#endif // ANYDSL_DEF_H
