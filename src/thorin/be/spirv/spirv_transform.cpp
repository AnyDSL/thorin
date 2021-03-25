#include <thorin/analyses/looptree.h>
#include "thorin/be/spirv/spirv.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"

#include <algorithm>
#include <thorin/analyses/domtree.h>

namespace thorin::spirv {

using Head = LoopTree<true>::Head;
using Base = LoopTree<true>::Base;
using Leaf = LoopTree<true>::Leaf;

struct StructuredLoop;

// Dispatch targets may dispatch to other dispatching nodes, and we can't actually create those until we know all their destinations,
// because their fn type takes a variant type with a case for each target. So we symbolically refer to these yet-to-be dispatch nodes via their loop
struct DispatchTarget {
    bool operator==(const DispatchTarget& rhs) const {
        return cont == rhs.cont &&entry == rhs.entry &&exit == rhs.exit;
    }

    Continuation* cont = nullptr;
    StructuredLoop* entry = nullptr;
    StructuredLoop* exit = nullptr;
};

struct RewireMe {
    RewireMe(Continuation* cont, int op) : cont(cont), op(op) {}
    Continuation* cont;
    int op;

    Continuation* backedge = nullptr;
    struct {
        std::vector<StructuredLoop*> exits;
        std::vector<StructuredLoop*> enters;
        Continuation* final_destination = nullptr;
    } non_local_jump;
};

// Represents one loop in the loop forest, that we then augment with a new loop header and epilogue, each dispatching
// respectively to nodes inside of the loop, and nodes outside of the loop once we break out
struct StructuredLoop {
    const Head* parent_head;
    const Head* head;
    const std::string name;

    std::vector<DispatchTarget> inner_destinations = {};
    std::vector<DispatchTarget> outer_destinations = {};

    // Created to serve as codegen helpers
    Continuation* new_header = nullptr;
    Continuation* new_epilogue = nullptr;
    Continuation* new_continue = nullptr;

    // Same as the inner/outer destinations, but entry/exits instead now point to the corresponding header/epilogue nodes
    std::vector<const Continuation*> header_destination_conts;
    std::vector<const Continuation*> epilogue_destination_conts;

    std::vector<RewireMe> rewire;
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

        StructuredLoop loop{parent, head, name};
        ctx.rewritten_loops.emplace(head, loop);

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

inline int index_of_destination(std::vector<DispatchTarget>& vec, DispatchTarget dest) {
    auto i = std::find(vec.begin(), vec.end(), dest);
    if (i == vec.end()) {
        assert(false && "Missing destination");
    } else return i - vec.begin();
}

inline std::vector<StructuredLoop*> get_path(ScopeContext& ctx, const Head* head) {
    std::vector<StructuredLoop*> path = {};
    assert(head != nullptr);
    while (head != nullptr) {
        auto* loop = &ctx.rewritten_loops[head];
        assert(loop != nullptr);
        path.emplace(path.begin(), loop);
        head = loop->parent_head;
    }
    return path;
}

inline void collect_dispatch_targets(World& world, ScopeContext& ctx, const Base* base, const Head* parent) {
    if (const Head* head = base->isa<Head>()) {
        for (auto& children : head->children()) {
            collect_dispatch_targets(world, ctx, &*children, head);
        }
    } else {
        const Leaf* leaf = base->as<Leaf>();
        auto cont = leaf->cf_node()->continuation();
        // For some nonsense reason, synthetic nodes created during scopes iteration leak in next iterations >:(
        if (cont->intrinsic() >= Intrinsic::SCFBegin && cont->intrinsic() < Intrinsic::SCFEnd)
            return;
        for (size_t i = 0; i < cont->num_ops(); i++) {
            auto def = cont->op(i);
            if (auto dest = def->isa_continuation()) {
                const Head* source_loop = *ctx.def2loop[cont];

                if (dest->intrinsic() == Intrinsic::Branch) {
                    continue;
                }

                assert(ctx.def2loop.find(dest) != ctx.def2loop.end());
                const Head* dest_loop = *ctx.def2loop[dest];

                if (source_loop != dest_loop) {
                    // We found a non-local jump
                    assert(ctx.rewritten_loops.find(source_loop) != ctx.rewritten_loops.end());
                    auto& loop = ctx.rewritten_loops[source_loop];

                    std::vector<StructuredLoop*> source_path = get_path(ctx, source_loop);
                    std::vector<StructuredLoop*> dest_path = get_path(ctx, dest_loop);
                    int bi = 0;
                    while (bi < std::min(source_path.size(), dest_path.size())) {
                        if (source_path[bi] == dest_path[bi])
                            bi++;
                        else break;
                    }

                    // The path is made out of a sequence of loops to break out of, and a sequence of loops to jump into
                    // these two sequences cannot be both empty (that wouldn't be a non-local jump then!)
                    std::vector<StructuredLoop*> leave;
                    std::vector<StructuredLoop*> enter;
                    for (int j = source_path.size() - 1; j >= bi; j--)
                        leave.emplace_back(source_path[j]);
                    for (int j = bi; j < dest_path.size(); j++)
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
                        assert(prev != nullptr);
                    }
                    for (auto loop : enter) {
                        DispatchTarget destination;
                        destination.entry = loop;

                        record_step(loop, destination);
                        last = 2;
                        prev = loop;
                        assert(prev != nullptr);
                    }

                    assert(last != 0);
                    DispatchTarget destination;
                    destination.cont = dest;
                    record_step(prev, destination);

                    RewireMe rewire(cont, i);
                    rewire.non_local_jump = {
                        std::move(leave),
                        std::move(enter),
                        dest
                    };
                    loop.rewire.emplace_back(rewire);
                    printf("nlj %s %d!\n", loop.name.c_str(), loop.rewire.size());
                } else if (source_loop == dest_loop && source_loop != nullptr) {
                    for (auto& head : source_loop->cf_nodes()) {
                        if (head->continuation() == dest) {
                            // We found a backedge
                            assert(ctx.rewritten_loops.find(source_loop) != ctx.rewritten_loops.end());
                            auto& loop = ctx.rewritten_loops[source_loop];

                            DispatchTarget destination;
                            destination.cont = dest;
                            record_destination(loop.inner_destinations, destination);

                            RewireMe rewire(cont, i);
                            rewire.backedge = head->continuation();
                            loop.rewire.emplace_back(rewire);
                            printf("backedge %s %d!\n", loop.name.c_str(), loop.rewire.size());
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

inline const Def* tuple_from_params(World& world, const ArrayRef<const Param*> params) {
    std::vector<const thorin::Def*> t;
    t.resize(params.size());
    for (size_t i = 0; i < params.size(); i++)
        t[i] = params[i];
    return world.tuple(t);
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
            const thorin::Continuation* target_cont;
            if (target.cont != nullptr) {
                target_cont = target.cont;
            } else if (target.entry != nullptr) {
                assert(target.entry->new_header != nullptr);
                target_cont = target.entry->new_header;
            } else {
                assert(false && "Header dispatches may not exit loops");
            }
            loop.header_destination_conts.push_back(target_cont);
            const thorin::FnType* target_type = target_cont->type();
            dest_types.emplace_back(dom_to_tuple(world, target_type));
        }
        auto variant_type = world.variant_type(loop.name + "_param", dest_types.size());
        for (size_t i = 0; i < dest_types.size(); i++)
            variant_type->set(i, dest_types[i]);
        auto fn_type = world.fn_type( { variant_type } );
        loop.new_header = world.continuation(fn_type, { loop.name + "_new_header"});
        loop.new_continue = world.continuation(fn_type, { loop.name + "_new_continue"});
    }
}

inline void create_epilogues(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        StructuredLoop& loop = ctx.rewritten_loops[head];

        if (head->num_cf_nodes() > 0) {
            // here, children epilogues need to know what they're jumping *out to*
            std::vector<const Type*> dest_types;
            for (auto& target : loop.outer_destinations) {
                const thorin::Continuation* target_cont;
                if (target.cont != nullptr) {
                    target_cont = target.cont;
                } else if (target.entry != nullptr) {
                    assert(target.entry->new_header != nullptr);
                    target_cont = target.entry->new_header;
                } else {
                    assert(target.exit != nullptr);
                    assert(target.exit->new_epilogue != nullptr);
                    target_cont = target.exit->new_epilogue;
                }
                loop.epilogue_destination_conts.push_back(target_cont);
                const thorin::FnType* target_type = target_cont->type();
                dest_types.emplace_back(dom_to_tuple(world, target_type));
            }
            auto variant_type = world.variant_type(loop.name + "_param", dest_types.size());
            for (size_t i = 0; i < dest_types.size(); i++)
                variant_type->set(i, dest_types[i]);
            auto fn_type = world.fn_type({variant_type});
            loop.new_epilogue = world.continuation(fn_type, {loop.name + "_new_epilogue"});
        }

        for (auto& children : head->children()) {
            create_epilogues(world, ctx, &*children);
        }
    }
}

// Finishes loop headers & epilogues, and re-wires backedges and non-local jumps to go through structured CF intrinsics
inline void rewire_loops(World& world, ScopeContext& ctx, const Base* base) {
    if (const Head* head = base->isa<Head>()) {
        assert(ctx.rewritten_loops.find(head) != ctx.rewritten_loops.end());
        auto& loop = ctx.rewritten_loops[head];

        if (head->num_cf_nodes() > 0) {
            loop.new_epilogue->structured_loop_merge(loop.new_header, loop.epilogue_destination_conts);
            loop.new_continue->structured_loop_continue(loop.new_header);
            loop.new_header->structured_loop_header(loop.new_epilogue, loop.new_continue, loop.header_destination_conts);
            printf("Loop %s!\n", loop.name.c_str());
            for (auto c : loop.header_destination_conts)
                printf("  header target: %s!\n", c->unique_name().c_str());
            for (auto c : loop.epilogue_destination_conts)
                printf("  epilogue target: %s!\n", c->unique_name().c_str());
        }

        for (auto& children : head->children()) {
            rewire_loops(world, ctx, &*children);
        }

        printf("rewires %s %d!\n", loop.name.c_str(), loop.rewire.size());
        for (auto& rewire : loop.rewire) {
            printf("rewire!\n");
            if (rewire.backedge != nullptr) {
                printf("handling BE!\n");
                DispatchTarget destination;
                destination.cont = rewire.backedge;
                auto variant_index = index_of_destination(loop.inner_destinations, destination);

                auto old_fn_type = rewire.backedge->type();
                auto wrapper = world.continuation(old_fn_type, {"synthetic_backedge_wrapper_to" + destination.cont->unique_name() });
                wrapper->attributes_.intrinsic = Intrinsic::SCFBackEdge;

                auto header_variant_type = loop.new_header->type()->op(0)->as<VariantType>();
                wrapper->jump(loop.new_continue, { world.variant(header_variant_type, tuple_from_params(world, wrapper->params()), variant_index) });

                rewire.cont->unset_op(rewire.op);
                rewire.cont->set_op(rewire.op, wrapper);
            } else {
                printf("handling NLJ!\n");

                auto& nlj = rewire.non_local_jump;

                auto old_fn_type = nlj.final_destination->type();
                auto wrapper = world.continuation(old_fn_type, {"synthetic_nlj_wrapper_to" + nlj.final_destination->unique_name() });
                wrapper->attributes_.intrinsic = Intrinsic::SCFNonLocalJump;

                printf("nlj = %s\n", wrapper->unique_name().c_str());
                printf("src = %s\n", rewire.cont->name().c_str());
                for (auto exit : nlj.exits)
                    printf("exit = %s\n", exit->name.c_str());
                for (auto enter : nlj.enters)
                    printf("enter = %s\n", enter->name.c_str());
                printf("dst = %s\n", nlj.final_destination->name().c_str());

                const Def* argument = tuple_from_params(world, wrapper->params());
                Continuation* first_jump = nullptr;

                DispatchTarget destination;
                destination.cont = nlj.final_destination;

                for (int i = nlj.enters.size() - 1; i >= 0; i--) {
                    StructuredLoop* loop_to_enter = nlj.enters[i];

                    auto variant_index = index_of_destination(loop_to_enter->inner_destinations, destination);
                    auto header_variant_type = loop_to_enter->new_header->type()->op(0)->as<VariantType>();
                    argument = world.variant(header_variant_type, argument, variant_index);

                    first_jump = loop_to_enter->new_header;
                    destination = {};
                    destination.entry = loop_to_enter;
                }

                for (int i = nlj.exits.size() - 1; i >= 0; i--) {
                    StructuredLoop* loop_to_exit = nlj.exits[i];

                    auto variant_index = index_of_destination(loop_to_exit->outer_destinations, destination);
                    auto header_variant_type = loop_to_exit->new_epilogue->type()->op(0)->as<VariantType>();
                    argument = world.variant(header_variant_type, argument, variant_index);

                    first_jump = loop_to_exit->new_epilogue;
                    destination = {};
                    destination.exit = loop_to_exit;
                }

                assert(first_jump != nullptr);
                wrapper->jump(first_jump, { argument });

                rewire.cont->unset_op(rewire.op);
                rewire.cont->set_op(rewire.op, wrapper);
            }
        }
    }
}

void CodeGen::structure_loops() {
    Scope::for_each(world(), [&](const Scope& scope) {
        ScopeContext context(scope);
        printf("top: %d\n", scope.has_free_params());

        const LoopTree<true>& looptree = context.cfa.f_cfg().looptree();
        tag_continuations(context, looptree.root(), nullptr);
        collect_dispatch_targets(world(), context, looptree.root(), nullptr);

        create_headers(world(), context, looptree.root());
        create_epilogues(world(), context, looptree.root());

        rewire_loops(world(), context, looptree.root());
        printf("done\n");
    });
}

template<typename Fn>
inline void iterate_ancestors(Continuation* cont, Fn fn) {
    ContinuationSet done;

    Continuations stack;
    stack.push_back(cont);
    while (!stack.empty()) {
        Continuation* top = stack.back();
        stack.pop_back();
        if (done.contains(top)) continue;
        if (top != cont) fn(top);
        done.insert(top);
        for (auto pred : top->preds()) {
            auto pred_cont = pred->isa_continuation();
            if (!pred_cont) continue;
            if (!done.contains(pred_cont)) {
                stack.push_back(pred_cont);
            }
        }
    }
}

void CodeGen::structure_flow() {
    Scope::for_each(world(), [&](const Scope& scope) {
        CFA cfa(scope);
        auto& post_dom_tree = cfa.b_cfg().domtree();

        for (auto def : scope.defs()) {
            if (auto cont = def->isa_continuation()) {
                auto cfn = cfa[cont];
                /*if (cont->callee() == world().branch()) {
                    printf("xd: %s\n", cont->unique_name().c_str());
                }*/

                //if (cont->intrinsic() >= Intrinsic::SCFBegin && cont->intrinsic() < Intrinsic::SCFEnd)
                //    continue;
                if (cont->preds().size() <= 1)
                    continue;

                printf("has more than 1 incoming branch: %s\n", cont->unique_name().c_str());

                /*for (auto pred : cont->preds()) {
                    auto& pred_dominators = post_dom_tree.children(cfa[cont]);
                    // TODO insert join nodes when this assert breaks
                    // TODO inspect postdom tree recursively
                    assert(std::find(pred_dominators.begin(), pred_dominators.end(), cfa[pred]) != pred_dominators.end());
                }*/
                iterate_ancestors(cont, [&](Continuation* ancestor) {
                    printf("  ancestor: %s\n", ancestor->unique_name().c_str());
                });
            }
        }
    });
}

}