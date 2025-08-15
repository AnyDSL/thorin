#ifndef THORIN_BE_EMITTER_H
#define THORIN_BE_EMITTER_H

#include<stack>
#include<queue>

namespace thorin {

template<class Value, class Type, class BB, class Child>
class Emitter {
private:
    constexpr const Child& child() const { return *static_cast<const Child*>(this); };
    constexpr Child& child() { return *static_cast<Child*>(this); };

    /// Internal wrapper for @p emit that checks and retrieves/puts the @c Value from @p defs_.
    Value emit_(const Def* def) {
        std::stack<const Def*> required_defs;
        std::queue<const Def*> todo;
        todo.push(def);

        while (!todo.empty()) {
            auto def = todo.front();
            todo.pop();
            if (defs_.lookup(def)) continue;

            if (auto memop = def->isa<MemOp>()) {
                todo.push(memop->mem());
                required_defs.push(memop->mem());
            } else if (auto extract = def->isa<Extract>()) {
                if (is_mem(extract)) {
                    todo.push(extract->agg());
                    required_defs.push(extract->agg());
                }
            }
        }

        while (!required_defs.empty()) {
            auto r = required_defs.top();
            required_defs.pop();
            emit_unsafe(r);
        }

        //auto place = def->no_dep() ? entry_ : scheduler_.smart(def);
        auto place = !scheduler_.scope().contains(def) ? entry_ : scheduler_.early(def);

        if (place) {
            auto& bb = cont2bb_[place];
            return child().emit_bb(bb, def);
        } else {
            return child().emit_constant(def);
        }
    }

protected:
    //@{
    /// @name default implementations
    void finalize(const Scope&) {}
    void finalize(Continuation*) {}
    //@}

    /// Recursively emits code. @c mem -typed @p Def%s return sth that is @c !child().is_valid(value) - this variant asserts in this case.
    Value emit(const Def* def) {
        auto res = emit_unsafe(def);
        assert(child().is_valid(res));
        return res;
    }

    /// As above but returning @c !child().is_valid(value) is permitted.
    Value emit_unsafe(const Def* def) {
        if (auto val = defs_.lookup(def)) return *val;
        if (auto cont = def->isa_nom<Continuation>()) {
            if (cont->has_body())
                queue_scope(cont);
            return defs_[cont] = child().emit_fun_decl(cont);
        }

        auto val = emit_(def);
        return defs_[def] = val;
    }

    void queue_scope(Continuation* entry) {
        if (entry->has_body())
            scopes_to_emit_.push(entry);
    }

    void emit_scopes(ScopesForest& forest) {
        while (!scopes_to_emit_.empty()) {
            auto entry = scopes_to_emit_.pop();
            Scope& scope = forest.get_scope(entry);
            emit_scope(scope, forest);
        }
    }

private:
    void emit_scope(const Scope& scope, ScopesForest& forest) {
        if (emitted_scopes_.contains(scope.entry()))
            return;
        emitted_scopes_.insert(scope.entry());
        scope_ = &scope;
        auto conts = schedule(scope);
        entry_ = scope.entry();
        //assert(entry_->is_returning());

        auto fct = child().prepare(scope);
        for (auto cont : conts) {
            if (cont->intrinsic() != Intrinsic::EndScope) child().prepare(cont, fct);
        }

        Scheduler new_scheduler(scope, forest);
        swap(scheduler_, new_scheduler);

        for (auto cont : conts) {
            if (cont->intrinsic() == Intrinsic::EndScope) continue;
            //assert(cont == entry_ || cont->is_basicblock());
            child().emit_epilogue(cont);
        }

        for (auto cont : conts) {
            if (cont->intrinsic() != Intrinsic::EndScope) child().finalize(cont);
        }
        child().finalize(scope);
        scope_ = nullptr;
    }

protected:
    Scheduler scheduler_;
    DefMap<Value> defs_;
    DefMap<Type> types_;
    ContinuationMap<BB> cont2bb_;
    unique_queue<ContinuationSet> scopes_to_emit_;
    DefSet emitted_scopes_;
    Continuation* entry_ = nullptr;
    const Scope* scope_ = nullptr;
};

}

#endif
