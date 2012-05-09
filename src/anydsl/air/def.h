#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <string>
#include <vector>

#include <boost/unordered_set.hpp>

#include "anydsl/air/airnode.h"

namespace anydsl {

class Use;
typedef boost::unordered_set<Use*> Uses;

class Def : public AIRNode {
private:

    virtual ~Def() { anydsl_assert(uses_.empty(), "there are still uses pointing to this def"); }

public:

    Def(IndexKind indexKind, const std::string& debug)
        : AIRNode(indexKind, debug) 
    {}

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

    const Uses& uses() const { return uses_; }

private:

    Uses uses_;
};

} // namespace anydsl

#endif // ANYDSL_DEF_H
