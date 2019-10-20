#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include "thorin/world.h"
#include "thorin/util/iterator.h"

namespace thorin {

class PassMan;

/// All Pass%es that want to be registered in the @p PassMan must implement this interface.
class Pass {
public:
    Pass(PassMan& man, size_t index)
        : man_(man)
        , index_(index)
    {}
    virtual ~Pass() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    size_t index() const { return index_; }
    //Scope& scope();
    World& world();
    ///@}
    /// @name hooks for the PassMan
    //@{
    virtual bool enter(Scope&);                                ///< TODO
    virtual void clear() {}                                    ///< Must clear all info in order to operate on the next @p Scope.
    virtual void retry() {}                                    ///< Setup all data for a retry.
    virtual const Def* rewrite(const Def* def) { return def; } ///< Rewrites @em structural @p Def%s.
    virtual void inspect(Def*) {}                              ///< Inspects a @em nominal @p Def when first encountering it.
    virtual void enter(Def*) {}                                ///< TODO
    /// Invoked after the @p PassMan has finished @p rewrite%ing a nominal.
    /// Return @c true if everthing's fine, @c false if you need a @p retry.
    virtual bool analyze(const Def*) { return true; }
    ///@}

private:
    PassMan& man_;
    size_t index_;
};

/**
 * An optimizer that combines several optimizations in an optimal way.
 * This is loosely based upon:
 * "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMan {
public:
    PassMan(World& world)
        : world_(world)
    {}

    World& world() const { return world_; }
    Scope& old_scope() const { return *old_scope_; }
    Scope& new_scope() const { return *new_scope_; }
    size_t num_passes() const { return passes_.size(); }
    template<class T = Def>
    T* new_entry() const { return new_entry_->template as<T>(); }
    template<class P, class... Args>
    PassMan& create(Args... args) { passes_.emplace_back(std::make_unique<P>(*this, passes_.size()), std::forward<Args>(args)...); return *this; }
    void run();
    Def* stub(Def* nom);

    const Def*  scope_map(const Def* old_def, const Def* new_def) { return  scope_old2new_[old_def] = new_def; }
    const Def* global_map(const Def* old_def, const Def* new_def) { return global_old2new_[old_def] = new_def; }
    const Def* lookup(const Def* old_def) {
        if (auto new_def =  scope_old2new_.lookup(old_def)) return *new_def;
        if (auto new_def = global_old2new_.lookup(old_def)) return *new_def;
        return old_def;
    }

private:
    bool run(Def*);
    bool analyze(const Def*);

    World& world_;
    Scope* old_scope_ = nullptr;
    Scope* new_scope_ = nullptr;
    Def* new_entry_ = nullptr;
    std::vector<std::unique_ptr<Pass>> passes_;
    std::vector<Pass*> scope_passes_;
    DefSet analyzed_;
    Nom2Nom stubs_;
    Def2Def  scope_old2new_;
    Def2Def global_old2new_;
};

inline World& Pass::world() { return man().world(); }

}

#endif
