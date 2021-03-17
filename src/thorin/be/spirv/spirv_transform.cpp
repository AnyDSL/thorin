#include <thorin/analyses/looptree.h>
#include "thorin/be/spirv/spirv.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"

#include <algorithm>

namespace thorin::spirv {

using Head = LoopTree<true>::Head;
using Base = LoopTree<true>::Base;
using Leaf = LoopTree<true>::Leaf;

struct StructuredLoop;

// Dispatch targets may dispatch to other dispatching nodes, and we can't actually create those until we know all their destinations,
// because their fn type takes a variant type with a case for each target. So we symbolically refer to these yet-to-be dispatch nodes via their loop
struct DispatchTarget {
    bool operator==(const DispatchTarget& rhs) const {
        return cont == rhs.cont &&
               entry == rhs.entry &&
               exit == rhs.exit;
    }

    Continuation* cont = nullptr;
    StructuredLoop* entry = nullptr;
    StructuredLoop* exit = nullptr;
};

// Represents one loop in the loop forest, that we then augment with a new loop header and epilogue, each dispatching
// respectively to nodes inside of the loop, and nodes outside of the loop once we break out
struct StructuredLoop {
    const Head* parent_head;
    const Head* head;
    const std::string name;

    std::vector<DispatchTarget> inner_destinations = {};
    std::vector<DispatchTarget> outer_destinations = {};

    Continuation* new_header = nullptr;
    Continuation* new_epilogue = nullptr;
};

struct ScopeContext {
    explicit ScopeContext(const Scope& scope)
    : cfa(scope)
    {}

    CFA cfa;
    ContinuationMap<const Head*> def2loop;
    std::unordered_map<const Head*, StructuredLoop> rewritten_loops;
};

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

/// Visits the forest and fills def2loop
inline void tag_continuations(ScopeContext& ctx, const Base* base, const Head* parent) {
    for (int i = 0; i < base->depth(); i++)
        printf(" ");
    if (auto* head = base->isa<Head>()) {
        printf("loop: header = ");
    }
    for (auto& node : base->cf_nodes()) {
        printf("%s ", node->continuation()->to_string().c_str());
    }
    printf("\n");

    if (const Head* head = base->isa<Head>()) {
        auto name = safe_name(head);

        for (auto& children : head->children()) {
            tag_continuations(ctx, &*children, head);
        }

        if (parent != nullptr) {
            StructuredLoop loop{parent, head, name};
            ctx.rewritten_loops.emplace(head, loop);
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

inline int record_destination(std::vector<DispatchTarget>& vec, DispatchTarget dest) {
    auto i = std::find(vec.begin(), vec.end(), dest);
    if (i == vec.end()) {
        vec.emplace_back(dest);
        return vec.size() - 1;
    } else return i - vec.begin();
}

inline std::vector<StructuredLoop*>&& get_path(ScopeContext& ctx, const Head* head) {
    std::vector<StructuredLoop*> path;
    do {
        if (head != nullptr) {
            auto* loop = &ctx.rewritten_loops[head];
            path.emplace(path.begin(), loop);
            head = loop->parent_head;
        } else {
            path.emplace(path.begin(), nullptr);
            head = nullptr;
        }
    } while (head != nullptr);
    return std::move(path);
}

inline void collect_dispatch_targets(World& world, ScopeContext& ctx, const Base* base, const Head* parent) {
    if (const Head* head = base->isa<Head>()) {
        for (auto& children : head->children()) {
            collect_dispatch_targets(world, ctx, &*children, head);
        }
    } else {
        const Leaf* leaf = base->as<Leaf>();
        auto cont = leaf->cf_node()->continuation();
        for (size_t i = 0; i < cont->num_ops(); i++) {
            auto def = cont->op(i);
            if (auto dest = def->isa_continuation()) {
                const Head* source_loop = *ctx.def2loop[cont];
                const Head* dest_loop = *ctx.def2loop[cont];

                if (source_loop != dest_loop) {
                    // We found a non-local jump

                    auto source_path = get_path(ctx, source_loop);
                    auto dest_path = get_path(ctx, dest_loop);
                    size_t bi = 0;
                    while (bi < std::min(source_path.size(), dest_path.size())) {
                        if (source_path[bi] == dest_path[bi])
                            bi++;
                        else break;
                    }

                    // The path is made out of a sequence of loops to break out of, and a sequence of loops to jump into
                    // these two sequences cannot be both empty (that wouldn't be a non-local jump then!)
                    std::vector<StructuredLoop*> leave;
                    std::vector<StructuredLoop*> enter;
                    for (size_t j = source_path.size() - 1; j >= bi; j--)
                        leave.emplace_back(source_path[j]);
                    for (size_t j = bi; j < dest_path.size(); j++)
                        enter.emplace_back(dest_path[j]);

                    // 0 = this is the first step of the path
                    // 1 = last step was to break out of a loop
                    // 2 = last step was to enter a loop
                    int last = 0;
                    StructuredLoop* prev;

                    auto record_step = [&](StructuredLoop* loop, DispatchTarget destination) {
                        if (last == 0) {
                            // nothing to do, this node isn't a dispatching one
                        } else {
                            if (last == 1)
                                record_destination(prev->outer_destinations, destination);
                            else
                                record_destination(prev->inner_destinations, destination);
                        }
                    };

                    for (auto loop : leave) {
                        DispatchTarget destination;
                        destination.exit = loop;

                        record_step(loop, destination);
                        last = 1;
                        prev = loop;
                    }
                    for (auto loop : enter) {
                        DispatchTarget destination;
                        destination.entry = loop;

                        record_step(loop, destination);
                        last = 2;
                        prev = loop;
                    }

                    assert(last != 0);
                    DispatchTarget destination;
                    destination.cont = dest;
                    record_step(prev, destination);
                } else if (source_loop == dest_loop && source_loop != nullptr) {
                    for (auto head : source_loop->cf_nodes()) {
                        if (head->continuation() == dest) {
                            // We found a backedge
                            auto loop = ctx.rewritten_loops[source_loop];

                            DispatchTarget destination;
                            destination.cont = dest;
                            record_destination(loop.inner_destinations, destination);
                            break;
                        }
                    }
                }
            }
        }
    }
}

inline const Type* dom_to_tuple(World& world, const thorin::FnType* fn_type) {
    std::vector<const thorin::Type*> t;
    t.resize(fn_type->num_ops());
    for (size_t i = 0; i < fn_type->num_ops(); i++)
        t[i] = fn_type->op(i);
    return world.tuple_type(t);
}

inline void create_headers(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        for (auto& children : head->children()) {
            create_headers(world, ctx, &*children);
        }

        if (head->num_cf_nodes() == 0)
            return;
        StructuredLoop& loop = ctx.rewritten_loops[head];

        // here, parent headers need to know what they're jumping *into*
        std::vector<const Type*> dest_types;
        for (auto& target : loop.inner_destinations) {
            const thorin::FnType* target_type;
            if (target.cont != nullptr) {
                target_type = target.cont->type();
            } else if (target.entry != nullptr) {
                assert(target.entry->new_header != nullptr);
                target_type = target.entry->new_header->type();
            } else {
                assert(false && "Header dispatches may not exit loops");
            }
            dest_types.emplace_back(dom_to_tuple(world, target_type));
        }
        auto variant_type = world.variant_type(loop.name + "_param", dest_types.size());
        for (size_t i = 0; i < dest_types.size(); i++)
            variant_type->set(i, dest_types[i]);
        auto fn_type = world.fn_type( { variant_type } );
        loop.new_header = world.continuation(fn_type, { loop.name + "_new_header"});
    }
}

inline void create_epilogues(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        StructuredLoop& loop = ctx.rewritten_loops[head];

        if (head->num_cf_nodes() > 0) {
            // here, children epilogues need to know what they're jumping *out to*
            std::vector<const Type*> dest_types;
            for (auto& target : loop.inner_destinations) {
                const thorin::FnType* target_type;
                if (target.cont != nullptr) {
                    target_type = target.cont->type();
                } else if (target.entry != nullptr) {
                    assert(target.entry->new_header != nullptr);
                    target_type = target.entry->new_header->type();
                } else {
                    assert(target.exit != nullptr);
                    assert(target.exit->new_epilogue != nullptr);
                    target_type = target.exit->new_epilogue->type();
                }
                dest_types.emplace_back(dom_to_tuple(world, target_type));
            }
            auto variant_type = world.variant_type(loop.name + "_param", dest_types.size());
            for (size_t i = 0; i < dest_types.size(); i++)
                variant_type->set(i, dest_types[i]);
            auto fn_type = world.fn_type({variant_type});
            loop.new_header = world.continuation(fn_type, {loop.name + "_new_epilogue"});
        }

        for (auto& children : head->children()) {
            create_epilogues(world, ctx, &*children);
        }
    }
}

// This creates the header/epilogue nodes for the loops
inline void augment_loops(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        auto& loop = ctx.rewritten_loops[head];
        auto name = safe_name(head);

        const thorin::Type* unified_heads_param;
        const thorin::VariantType* header_variant_type = nullptr;

        auto fn_type = world.fn_type( { unified_heads_param } );

        // Handle internal backedges: re-wire them to go through header
        for (size_t header_index = 0; header_index < base->num_cf_nodes(); header_index++) {
            auto& header_node = base->cf_nodes()[header_index];

            const Head* dest_loop = *ctx.def2loop[header_node->continuation()];
            for (auto& pred : header_node->continuation()->preds()) {
                const Head* src_loop = *ctx.def2loop[pred];
                printf("%s -> %s\n", safe_name(src_loop).c_str(), safe_name(dest_loop).c_str());

                if (src_loop == dest_loop) {
                    // get_or_create_destination(loop.inner_destinations, header_node->continuation());

                    for (int i = 0; i < pred->num_ops(); i++) {
                        if (pred->op(i) == header_node->continuation()) {
                            pred->unset_op(i);

                            // Oops except we can't do that! The new header has a mismatched signature, so we must go through a synthetic wrapper
                            //pred->set_op(i, loop.new_header);

                            auto wrapper = world.continuation(fn_type, {"synthetic_backedge_wrapper"});
                            wrapper->jump(loop.new_header, { world.variant(header_variant_type, world.tuple(pred->args()), header_index) });
                            pred->set_op(i, wrapper);
                        }
                    }
                }
            }
        }

        for (auto& children : head->children()) {
            augment_loops(world, ctx, &*children);
        }
    }
}

void CodeGen::structure_loops() {
    Scope::for_each(world(), [&](const Scope& scope) {
        ScopeContext context(scope);

        const LoopTree<true>& looptree = context.cfa.f_cfg().looptree();
        tag_continuations(context, looptree.root(), nullptr);
        collect_dispatch_targets(world(), context, looptree.root(), nullptr);

        create_headers(world(), context, looptree.root());
        create_epilogues(world(), context, looptree.root());

        augment_loops(world(), context, looptree.root());
    });
}

void CodeGen::structure_flow() {
    // TODO
}

}