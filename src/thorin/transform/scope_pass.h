#if 0

#ifndef THORIN_TRANSFORM_SCOPE_PASS_H
#define THORIN_TRANSFORM_SCOPE_PASS_H

#include "thorin/analyses/scope.h"

namespace thorin {

class Scope;

class ScopePass {
public:
    ScopePass(const Scope& scope)
        : scope_(scope)
    {}

    virtual ~ScopePass() {}
    virtual void run(const Scope&) = 0;
    virtual void run(World&);

private:
};

}

#endif
#endif
