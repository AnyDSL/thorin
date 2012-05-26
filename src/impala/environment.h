#ifndef IMPALA_ENVIROMENT_H
#define IMPALA_ENVIROMENT_H

#include <map>
#include <set>
#include <stack>

#include "anydsl/symbol.h"

#include "impala/value.h"

namespace anydsl {
    class Def;
    class Type;
}

namespace impala {

//------------------------------------------------------------------------------

/*
 * TODO implement sth like a stack manager
 */

/** 
 * @brief Keeps track of a current Type of a \p Symbol in a stack of scopes.
 */
class Environment {
public:

    Environment();
    ~Environment();

    /** 
     * @brief Looks up the current definition of \p sym.
     * 
     * @param sym The \p Symbol to look up.
     * 
     * @return Returns 0 on failure.
     */
    const anydsl::Type* lookup(const anydsl::Symbol sym);

    /** 
     * @brief Pushes a new Type on the internal \p TypeStack.
     * 
     * If sym already has a definition in this scope an assertion is raised.
     * use \p clash in order to check this.
     *
     * @param sym The \p Symbol to insert.
     */
    void insert(const anydsl::Symbol sym, const anydsl::Type* type);

    /** 
     * @brief Checks whether there already exists a \p Symbol \p sym in the \em current scope.
     * 
     * @param sym The \p Symbol to check.
     * 
     * @return The current mapping if the lookup succeeds, 0 otherwise.
     */
    const anydsl::Type* clash(const anydsl::Symbol sym) const;

    /// Open a new scope.
    void pushScope();
    /// Discard current scope.
    void popScope();

private:

    typedef std::stack<const anydsl::Type*> TypeStack;
    typedef std::map<const anydsl::Symbol, TypeStack*, anydsl::Symbol::FastLess> Scope;
    typedef std::stack<Scope> ScopeStack;
    typedef std::map<const anydsl::Symbol, TypeStack*, anydsl::Symbol::FastLess> Sym2TypeStack;

    Scope& curScope() { return scopeStack_.top(); }
    const Scope& curScope() const { return scopeStack_.top(); }

    ScopeStack scopeStack_;
    Sym2TypeStack sym2typestack_;

#ifndef NDEBUG
    mutable int refcounter_;
#endif
};

} // namespace impala

#endif // IMPALA_ENVIROMENT_H
