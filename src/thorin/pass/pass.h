#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/bitset.h"

namespace thorin {

class PassMan;

/// All @p Pass%es that want to be registered in the @p PassMan must implement this interface.
class Pass {
public:
    Pass(PassMan& man, size_t index, const std::string& name)
        : man_(man)
        , index_(index)
        , name_(name)
    {}
    virtual ~Pass() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    size_t index() const { return index_; }
    World& world();
    ///@}
    /// @name hooks for the PassMan
    //@{
    virtual bool scope(Def* nom) = 0;                          ///< Should enter scope with entry @p nom?
    virtual bool enter(Def* nom) = 0;                          ///< Should enter @p nom within current scope?
    virtual Def* inspect(Def* def) { return def; }             ///< Inspects a @em nominal @p Def when first encountering it.
    virtual const Def* rewrite(const Def* def) { return def; } ///< Rewrites @em structural @p Def%s.
    virtual bool analyze(const Def*) { return true; }          ///< Return @c true if everthing's fine, @c false if you need a @p retry.
    virtual void retry() {}                                    ///< Setup all data for a retry.
    virtual void clear() {}                                    ///< Must clear all info in order to operate on the next @p Scope.
    ///@}

private:
    PassMan& man_;
    size_t index_;
    std::string name_;
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

    /// @name create and run
    //@{
    /// Add @p Pass to this @p PassMan.
    template<class P, class... Args>
    PassMan& create(Args... args) {
        passes_.emplace_back(std::make_unique<P>(*this, passes_.size()), std::forward<Args>(args)...);
        return *this;
    }
    void run(); ///< Run all registered @p Pass%es on the whole @p world.
    //@}
    /// @name getters
    //@{
    World& world() const { return world_; }
    size_t num_passes() const { return passes_.size(); }
    template<class T = Def> T* cur_nom() const { return cur_nom_->template as<T>(); }
    //@}
    /// @name map either scope-wide or globally
    const Def*  local_map(const Def* old_def, const Def* new_def) { assert(old_def != new_def); return  local_.map[old_def] = new_def; }
    const Def* global_map(const Def* old_def, const Def* new_def) { assert(old_def != new_def); return global_.map[old_def] = new_def; }
    const Def* lookup(const Def* old_def) {
        // TODO path compression
        if (auto new_def =  local_.map.lookup(old_def)) return lookup(*new_def);
        if (auto new_def = global_.map.lookup(old_def)) return lookup(*new_def);
        return old_def;
    }
    Def* lookup(Def* old_nom) { return lookup(const_cast<const Def*>(old_nom))->as_nominal(); }
    template<class T> T* new2old(T* new_nom) { if (auto old_nom = new2old_.lookup(new_nom)) return (*old_nom)->template as<T>(); return new_nom->template as<T>(); }
    template<class T> T* new2old(T* new_nom, T* old_nom) { return (new2old_[new_nom] = old_nom)->template as<T>(); }
    //@}
    /// @name misc
    //@{
    bool outside(Def* nom); /// Tests whether @p nom%inal is outside current scope.
    //@}

private:
    bool scope();
    const Def* rewrite(const Def*);
    bool analyze(const Def*);

    World& world_;
    std::vector<std::unique_ptr<Pass>> passes_;
    Nom2Nom stubs_;
    Nom2Nom new2old_;
    Def* old_entry_ = nullptr;
    Def* new_entry_ = nullptr;
    Def* cur_nom_ = nullptr;

    struct Global {
        Def2Def map;
        unique_queue<NomSet> noms;
    } global_;

    struct Local : public Global {
        Scope* old_scope = nullptr;
        std::vector<Pass*> passes;
        std::vector<Pass*> cur_passes;
        NomSet free;
        DefSet rewritten;
        DefSet analyzed;

        void clear(Scope& s) {
            old_scope = &s;
            passes.clear();
            cur_passes.clear();
            map.clear();
            noms.clear();
            free.clear();
            rewritten.clear();
            analyzed.clear();
        }
    } local_;
};

inline World& Pass::world() { return man().world(); }

}

#endif
