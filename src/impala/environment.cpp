#include <impala/environment.h>

#include "anydsl/util/assert.h"
#include "anydsl/util/foreach.h"

using anydsl::Symbol;
using anydsl::Type;

namespace impala {

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

/*
 * constructor and destructor
 */

Environment::Environment()
#ifndef NDEBUG
    : refcounter_(0)
#endif
{
    pushScope();
}

Environment::~Environment() {
    while (!scopeStack_.empty())
        popScope();

#ifndef NDEBUG
    anydsl_assert(refcounter_ == 0, "memory leak");
#endif
}

const Type* Environment::lookup(const Symbol sym) {
    Sym2TypeStack::iterator i = sym2typestack_.find(sym);
    if (i != sym2typestack_.end())
        return i->second->top();
    else
        return 0;
}

void Environment::insert(const Symbol sym, const Type* type) {
    anydsl_assert(clash(sym) == 0, "must not be found");

    // create stack if necessary
    Sym2TypeStack::iterator i = sym2typestack_.find(sym);
    if (i == sym2typestack_.end()) {
        // isn't C++ beautiful? We should switch to C++11 just because of auto
        std::pair<Sym2TypeStack::iterator, bool> p =
            sym2typestack_.insert(std::make_pair(sym, new TypeStack()));
#ifndef NDEBUG
        ++refcounter_;
#endif
        i = p.first;
    }

    // get pointer to stack linked to sym and push new Type
    TypeStack* stack = i->second;
    stack->push(type);

    // remember that sym has an entry in current scope
    curScope()[sym] = stack;
}

const Type* Environment::clash(const Symbol sym) const {
    Scope::const_iterator i = curScope().find(sym);
    if (i != curScope().end())
        return i->second->top();

    return 0;
}

void Environment::popScope() {
    FOREACH(& i, curScope()) {
        const Symbol& sym = i.first;
        TypeStack* stack = i.second;

        anydsl_assert(!stack->empty(), "must have at least on element");
        stack->pop();

        if (stack->empty()) {
            sym2typestack_.erase(sym);
            delete stack;
#ifndef NDEBUG
            --refcounter_;
#endif
        }
    }

    scopeStack_.pop();
}

void Environment::pushScope() {
    scopeStack_.push(Scope());
}

} // namespace impala
