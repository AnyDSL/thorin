#if 0
#include "thorin/transform/live.h"

namespace thorin {

//------------------------------------------------------------------------------

bool is_equal_map(const DefMap<DefSet>& lhs, const DefMap<DefSet>& rhs)  {
    if(lhs.size() != rhs.size()) {
        return false;
    }
    for(auto& p : lhs) {
        // visit each DefSet stored in lhs and see if rhs also has
        // *something* stored for the corresponding key
        auto it = rhs.find(p.first);
        if(it != rhs.end()) {
            if(p.second.size() != it->second.size()) {
                return false;
            }
            // exhaustively search through the set of lhs
            for(auto& e : p.second) {
                if(!it->second.contains(e))
                    return false;
            }
        }
        else
          return false;
    }
    return true;
}
//------------------------------------------------------------------------------

template<class F>
void reverse_post_order_world(World& world, F&& func)
{
    for(auto lam : world.externals()) {
        assert(!lam->is_empty() && "external must not be empty");

        auto walk = [&lam,&func,colors = DefMap<int>()]() mutable {
            auto walk_impl = [&func,&colors](Lam* lam, auto& walk_ref) -> void {
                if (lam->is_empty())
                    return;
                Scope scope(lam);

                colors[lam] = 0;

                unique_queue<DefSet> def_queue;
                for (auto def : scope.free())
                    def_queue.push(def);

                while (!def_queue.empty()) {
                    auto def = def_queue.pop();
                    if (auto lam = def->isa_nominal<Lam>())
                        walk_ref(lam, walk_ref);
                    else {
                        for (auto op : def->ops())
                            def_queue.push(op);
                    }
                }
                // visit element on leave
                for(auto* def : scope.defs())
                    func(def);
                colors[lam] = 1;
            };
            walk_impl(lam, walk_impl);
        };
        walk();
    }
}

//------------------------------------------------------------------------------

Live::Live(World& world)
    : world(world)
{ analyze(); }

void Live::analyze()
{
    // our analysis is very fine-grained in the sense that we consider all Defs basically
    // as basic blocks. This is needed for stack-heavy lams, otherwise, we don't detect
    // any interferences...
    DefMap<DefSet> in, out, use, def;
    reverse_post_order_world(world, [&in,&out,&use,&def](const Def* d) {
        in[d] = {};
        out[d] = {};

        if(auto par = d->isa<Param>()) {
            def[d].insert(par);
        }
        else if(auto mem = d->isa<MemOp>()) {
            def[d].insert(mem->out_mem());
        }
        for(auto& u : d->uses()) {
            use[d].insert(u);
        }
    });
    DefMap<DefSet> tmp_in, tmp_out;
    do {
        reverse_post_order_world(world, [&tmp_in,&in,&tmp_out,&out,&def,&use](const Def* n) {
            tmp_in[n] = in[n];
            tmp_out[n] = out[n];

            // in[n] <- use[n] \/ (out[n] \ def[n])
            in[n] = use[n];

            auto out_with_no_def = out[n];
            for(auto to_find : def[n]) {
                auto it = out_with_no_def.find(to_find);
                if(it != out_with_no_def.end())
                    out_with_no_def.erase(it);
            }

            in[n].insert(out_with_no_def.begin(), out_with_no_def.end());

            for(auto& s : n->uses()) {
                out[n].insert(s);
            }
        });
    } while(!is_equal_map(tmp_in, in) || !is_equal_map(tmp_out, out));
    tmp_in.clear();
    tmp_out.clear();

    DefMap<DefSet> edges;
    reverse_post_order_world(world, [&edges,&def,&out](const Def* n) {
        // We add an edge if v and w both are alive in at least one program point simultaneously
        for(auto a : def[n]) {
            for(auto b : out[n]) {
                // add edge from a to b
                edges[a].emplace(b);
            }
        }
    });

    std::cout << "digraph ig {\n";

    for(auto& p : edges) {
        for(auto& b : p.second) {
            std::cout << p.first->unique_name() << " -> " << b->unique_name() << ";\n";
        }
    }
    std::cout << "}\n";

}

//------------------------------------------------------------------------------

}
#endif
