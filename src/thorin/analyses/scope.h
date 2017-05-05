#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/continuation.h"
#include "thorin/util/array.h"
#include "thorin/util/iterator.h"
#include "thorin/util/stream.h"

namespace thorin {

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class CFA;
class CFNode;

/**
 * A @p Scope represents a region of @p Continuation%s which are live from the view of an @p entry @p Continuation.
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 */
class Scope : public Streamable {
public:
    class Node {
    public:
        Node(Continuation* continuation, const Node* parent, int depth)
            : continuation_(continuation)
            , parent_(parent)
            , depth_(depth)
        {}

        Continuation* continuation() const { return continuation_; }
        const Node* parant() const { return parent_; }
        ArrayRef<std::unique_ptr<const Node>> children() const { return children_; }
        int depth() const { return depth_; }

    private:
        const Node* bear(Continuation* continuation) const {
            children_.emplace_back(std::make_unique<const Node>(continuation, this, depth() + 1));
            return children_.back().get();
        }

        Continuation* continuation_;
        const Node* parent_;
        mutable std::vector<std::unique_ptr<const Node>> children_;
        int depth_;

        friend class TreeBuilder;
    };

    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Continuation* entry);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    const Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Continuation* entry() const { return top_down().front(); }
    Continuation* exit() const { return top_down().back(); }
    size_t size() const { return top_down().size(); }
    //@}

    //@{ traversal
    ArrayRef<Continuation*> top_down() const { return top_down_; }
    auto bottom_up() const { return range(top_down().rbegin(), top_down().rend()); }
    //@}

    //@{ get Def%s contained in this Scope
    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    bool inner_contains(Continuation* continuation) const { return continuation != entry() && contains(continuation); }
    bool inner_contains(const Param* param) const { return inner_contains(param->continuation()); }
    //@}

    //@{ get CFG
    const CFA& cfa() const;
    const CFNode* cfa(Continuation*) const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    template<bool forward> const CFG<forward>& cfg() const;
    //@}

    //@{ dump
    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override;  ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;               ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                         ///< Dumps thorin to a file with an auto-generated file name.
    //@}

    /**
     * Transitively visits all @em reachable Scope%s in @p world that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Continuation%s without body.
     */
    template<bool elide_empty = true>
    static void for_each(const World& world, std::function<void(Scope&)>);

private:
    void run(Continuation* entry);
    void nesting();

    World& world_;
    DefSet defs_;
    std::vector<Continuation*> top_down_;
    mutable std::unique_ptr<const CFA> cfa_;

    friend class TreeBuilder;
};

template<> inline const CFG< true>& Scope::cfg< true>() const { return f_cfg(); }
template<> inline const CFG<false>& Scope::cfg<false>() const { return b_cfg(); }

}

#endif
