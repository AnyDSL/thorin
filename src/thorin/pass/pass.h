#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <map>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

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
    const std::string& name() const { return name_; }
    World& world();
    ///@}
    /// @name hooks for the PassMan
    //@{
    virtual bool enter(Def* nom) = 0;                          ///< Should enter @p nom within current scope?
    virtual Def* inspect(Def* def) { return def; }             ///< Inspects a @em nominal @p Def when first encountering it.
    virtual const Def* rewrite(const Def* def) { return def; } ///< Rewrites @em structural @p Def%s.
    virtual bool analyze(const Def*) { return true; }          ///< Return @c true if everthing's fine, @c false if you need a @p retry.
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

private:
    Def* rewrite(Def*);
    const Def* rewrite(const Def*, std::pair<const ReplArray, Def2Def>&);
    bool analyze(const Def*);

    World& world_;
    std::vector<std::unique_ptr<Pass>> passes_;
    Nom2Nom new2old_;
    Def* cur_nom_ = nullptr;
};

inline World& Pass::world() { return man().world(); }

}

#endif
