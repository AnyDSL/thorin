#include "thorin/analyses/scope.h"

#include <algorithm>
#include <fstream>

#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

Scope::Scope(Continuation* entry)
    : world_(entry->world())
{
    run(entry);
}

Scope::~Scope() {}

const Scope& Scope::update() {
    auto e = entry();
    top_down_.clear();
    defs_.clear();
    cfa_ = nullptr;
    run(e);
    return *this;
}

void Scope::run(Continuation* entry) {
    std::queue<const Def*> queue;

    auto enqueue = [&] (const Def* def) {
        if (defs_.insert(def).second) {
            queue.push(def);

            if (auto continuation = def->isa_continuation()) {
                top_down_.push_back(continuation);

                for (auto param : continuation->params()) {
                    auto p = defs_.insert(param);
                    assert_unused(p.second);
                    queue.push(param);
                }
            }
        }
    };

    enqueue(entry);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (def != entry) {
            for (auto use : def->uses())
                enqueue(use);
        }
    }

    enqueue(world().end_scope());
}

const CFA& Scope::cfa() const { return lazy_init(this, cfa_); }
const CFNode* Scope::cfa(Continuation* continuation) const { return cfa()[continuation]; }
const F_CFG& Scope::f_cfg() const { return cfa().f_cfg(); }
const B_CFG& Scope::b_cfg() const { return cfa().b_cfg(); }

template<bool elide_empty>
void Scope::for_each(const World& world, std::function<void(Scope&)> f) {
    ContinuationSet done;
    std::queue<Continuation*> queue;

    auto enqueue = [&] (Continuation* continuation) {
        const auto& p = done.insert(continuation);
        if (p.second)
            queue.push(continuation);
    };

    for (auto continuation : world.externals()) {
        assert(!continuation->empty() && "external must not be empty");
        enqueue(continuation);
    }

    while (!queue.empty()) {
        auto continuation = pop(queue);
        if (elide_empty && continuation->empty())
            continue;
        Scope scope(continuation);
        f(scope);

        for (auto n : scope.f_cfg().reverse_post_order()) {
            for (auto succ : n->continuation()->succs()) {
                if (!scope.contains(succ))
                    enqueue(succ);
            }
        }
    }
}

template void Scope::for_each<true> (const World&, std::function<void(Scope&)>);
template void Scope::for_each<false>(const World&, std::function<void(Scope&)>);

std::ostream& Scope::stream(std::ostream& os) const { return schedule(*this).stream(os); }
void Scope::write_thorin(const char* filename) const { return schedule(*this).write_thorin(filename); }
void Scope::thorin() const { schedule(*this).thorin(); }

}
