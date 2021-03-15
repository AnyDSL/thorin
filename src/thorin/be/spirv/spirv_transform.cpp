#include <thorin/analyses/looptree.h>
#include "thorin/be/spirv/spirv.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"

#include <algorithm>

namespace thorin::spirv {

using Head = LoopTree<true>::Head;
using Base = LoopTree<true>::Base;
using Leaf = LoopTree<true>::Leaf;

struct StructuredLoop {
    const Head* old_head;
    const std::string name;
    Continuation* new_header;
    Continuation* inner_dispatch;
    Continuation* outer_dispatch;

    std::vector<Continuation*> inner_destinations;
    std::vector<Continuation*> outer_destinations;
};

inline int get_or_create_destination(std::vector<Continuation*>& vec, Continuation* destination) {
    if (auto i = std::find(vec.begin(), vec.end(), destination); i != vec.end())
        return i - vec.begin();
    vec.emplace_back(destination);
    return vec.size() - 1;
}

struct ScopeContext {
    explicit ScopeContext(const Scope& scope)
    : cfa(scope)
    {}

    CFA cfa;
    ContinuationMap<const Head*> def2loop;
    std::unordered_map<const Head*, StructuredLoop> rewritten_loops;
};

/// Visits the forest and fills def2loop
inline void tagContinuations(ScopeContext& ctx, const Base* base, const Head* parent) {
    for (int i = 0; i < base->depth(); i++)
        printf(" ");
    if (auto* head = base->isa<Head>()) {
        printf("loop: header = ");
    }
    for (auto& node : base->cf_nodes()) {
        printf("%s ", node->continuation()->to_string().c_str());
    }
    printf("\n");

    if (auto* head = base->isa<Head>()) {
        for (auto& children : head->children()) {
            tagContinuations(ctx, &*children, head);
        }
    } else if(auto* leaf = base->isa<Leaf>()) {
        for (auto& node : base->cf_nodes()) {
            auto[i, result] = ctx.def2loop.emplace(node->continuation(), parent);
            assert(result);
        }
    } else {
        assert(false);
    }
}

inline std::string safe_name(const Head* head) {
    if (head == nullptr || head->is_root()) {
        return "root";
    } else {
        std::stringstream s;
        s << "loop_";
        for (auto& node : head->cf_nodes()) {
            s << node->continuation()->to_string();
            s << "_";
        }
        return s.str();
    }
}

inline void augmentLoops(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        StructuredLoop loop {
            head,
            safe_name(head),
            world.continuation({ safe_name(head) + "_header"}),
            world.continuation({ safe_name(head) + "_inner_dispatch"}),
            world.continuation({ safe_name(head) + "_outer_dispatch"}),
            {},
            {},
        };

        for (auto& header_node : base->cf_nodes()) {
            const Head* dest_loop = *ctx.def2loop[header_node->continuation()];
            for (auto& pred : header_node->continuation()->preds()) {
                const Head* src_loop = *ctx.def2loop[pred];
                printf("%s -> %s\n", safe_name(src_loop).c_str(), safe_name(dest_loop).c_str());

                // Backedge !
                if (src_loop == dest_loop) {
                    get_or_create_destination(loop.inner_destinations, header_node->continuation());

                    // point backedge to loop header
                    for (int i = 0; i < pred->num_ops(); i++) {
                        if (pred->op(i) == header_node->continuation()) {
                            pred->unset_op(i);
                            pred->set_op(i, loop.new_header);
                        }
                    }
                }
            }
        }

        ctx.rewritten_loops.emplace(head, loop);

        for (auto& children : head->children()) {
            augmentLoops(world, ctx, &*children);
        }
    }
}

void CodeGen::structure_loops() {
    Scope::for_each(world(), [&](const Scope& scope) {
        ScopeContext context(scope);

        const LoopTree<true>& looptree = context.cfa.f_cfg().looptree();
        tagContinuations(context, looptree.root(), nullptr);

        augmentLoops(world(), context, looptree.root());
    });
}

void CodeGen::structure_flow() {
    // TODO
}

}