#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/indexset.h"

namespace thorin {

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class CFA;
class InNode;

/**
 * @brief A @p Scope represents a region of @p Lambda%s which are live from the view of an @p entry @p Lambda.
 *
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 * Use @p lambdas() to retrieve a vector of @p Lambda%s in this @p Scope.
 * @p entry() will be first, @p exit() will be last.
 * @warning All other @p Lambda%s are in no particular order.
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

    const Scope& update();
    ArrayRef<Lambda*> lambdas() const { return lambdas_; }
    Lambda* operator [] (size_t i) const { return lambdas_[i]; }
    Lambda* entry() const { return lambdas().front(); }
    Lambda* exit() const { return lambdas().back(); }
    ArrayRef<Lambda*> body() const { return lambdas().skip_front(); } ///< Like @p lambdas() but without \p entry()

    const DefSet& in_scope() const { return in_scope_; }
    /// deprecated.
    bool _contains(Def def) const { return in_scope_.contains(def); }
    // TODO fix this: recursion/parameters
    bool outer_contains(Lambda* lambda) const { return lambda->find_scope(this) != nullptr; }
    bool outer_contains(const Param* param) const { return outer_contains(param->lambda()); }
    bool inner_contains(Lambda* lambda) const { return lambda != entry() && outer_contains(lambda); }
    bool inner_contains(const Param* param) const { return inner_contains(param->lambda()); }
    size_t index(Lambda* lambda) const {
        if (auto info = lambda->find_scope(this))
            return info->index;
        return size_t(-1);
    }
    uint32_t id() const { return id_; }
    size_t size() const { return lambdas_.size(); }
    World& world() const { return world_; }
    void dump() const;
    const CFA& cfa() const;
    const InNode* cfa(Lambda*) const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    template<bool forward> const CFG<forward>& cfg() const;
    void verify() const;

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return lambdas().begin(); }
    const_iterator end() const { return lambdas().end(); }

    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(Scope&)>);

private:
    void cleanup();
    void run(Lambda* entry);

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
};

template<> inline const CFG< true>& Scope::cfg< true>() const { return f_cfg(); }
template<> inline const CFG<false>& Scope::cfg<false>() const { return b_cfg(); }

}

#endif
