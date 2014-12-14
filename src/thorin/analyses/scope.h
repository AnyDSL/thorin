#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/indexset.h"

namespace thorin {

class CFA;
class Scope;

/**
 * A \p Scope represents a region of \p Lambda%s which are live from the view of an \p entry \p Lambda.
 * Transetively, all user's of the \p entry's parameters are pooled into this \p Scope.
 */
class Scope {
public:
    template<class Value>
    using Map = IndexMap<Scope, Lambda*, Value>;
    using Set = IndexSet<Scope, Lambda*>;

    Scope(const Scope&) = delete;
    Scope& operator= (Scope) = delete;

    explicit Scope(Lambda* entry);
    ~Scope();

    /// All lambdas within this scope in reverse post-order.
    ArrayRef<Lambda*> lambdas() const { return lambdas_; }
    Lambda* operator [] (size_t i) const { return lambdas_[i]; }
    Lambda* entry() const { return lambdas().front(); }
    Lambda* exit() const { return lambdas().back(); }
    /// Like \p lambdas() but without \p entry()
    ArrayRef<Lambda*> body() const { return lambdas().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool _contains(Def def) const { return in_scope_.contains(def); }
    bool contains(Lambda* lambda) const { return lambda->find_scope(this) != nullptr; }
    bool contains(const Param* param) const { return param->lambda()->find_scope(this) != nullptr; }
    size_t index(Lambda* lambda) const { 
        if (auto info = lambda->find_scope(this))
            return info->index; 
        return size_t(-1);  
    }
    uint32_t id() const { return id_; }
    size_t size() const { return lambdas_.size(); }
    World& world() const { return world_; }
    void dump() const;
    const CFA* cfa() const;

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return lambdas().begin(); }
    const_iterator end() const { return lambdas().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return lambdas().rbegin(); }
    const_reverse_iterator rend() const { return lambdas().rend(); }

    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(const Scope&)>);

private:
    static bool is_candidate(Def def) { return def->candidate_ == candidate_counter_; }
    static void set_candidate(Def def) { def->candidate_ = candidate_counter_; }
    static void unset_candidate(Def def) { assert(is_candidate(def)); --def->candidate_; }

    void identify_scope(Lambda* entry);
    void build_in_scope();

    World& world_;
    DefSet in_scope_;
    uint32_t id_;
    std::vector<Lambda*> lambdas_;
    mutable AutoPtr<const CFA> cfa_;

    static uint32_t candidate_counter_;
    static uint32_t id_counter_;

    template<bool> friend class ScopeView;
};

}

#endif
