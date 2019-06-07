#include "thorin/transform/live.h"

namespace thorin {

//------------------------------------------------------------------------------

bool is_unequal_map(const DefMap<DefSet>& lhs, const DefMap<DefSet>& rhs)  {
    for(auto& p : lhs) {
        // visit each DefSet stored in lhs and see if rhs also has
        // *something* stored for the corresponding key
        auto it = rhs.find(p.first);
        if(it != rhs.end()) {
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
    DefMap<int> colors;
    // compute reverse_post_ordering
    for(auto bb : world.externals()) {
        const auto walk = [&func,&bb,&colors]() {
            auto walk_impl = [&func,&colors](const Lam* bb, auto& walk_ref) -> void {
                colors[bb] = 0;
                for(auto& succ : bb->succs()) {
                    if(auto it = colors.find(succ); it == colors.end()) {
                        walk_ref(succ, walk_ref);
                    }
                }
                // visit element on leave
                func(bb);
                colors[bb] = 1;
            };
            walk_impl(bb, walk_impl);
        };
        walk();
    }
}

//------------------------------------------------------------------------------

Live::Live(World& world)
    : world(world)
{  }

void Live::analyze()
{
    DefMap<DefSet> in, out, use, def;
    reverse_post_order_world(world, [&in,&out,&use,&def](const Lam* bb) {
        // populate in and out sets
        for(auto& op : bb->body()->ops()) {
            if(op->isa_nominal<Lam>()) {
                in[op] = {};
                out[op] = {};
            }

            // parameters are the only source for new variables
            if(auto par = op->isa<Param>()) {
                def[op].insert(par);
            }
            for(auto& u : op->uses()) {
                if(op->is_value()) {
                    use[op].insert(u);
                }
            }
        }
    });
    DefMap<DefSet> tmp_in, tmp_out;
    do {
        reverse_post_order_world(world, [&tmp_in,&in,&tmp_out,&out,&def,&use](const Lam* n) {
            tmp_in[n] = in[n];
            tmp_out[n] = out[n];

            // in[n] <- use[n] \/ (out[n] \ def[n])
            in[n] = use[n];

            auto out_with_no_def = out[n];
            out_with_no_def.erase(def[n].begin(), def[n].end());

            in[n].insert(out_with_no_def.begin(), out_with_no_def.end());

            for(auto& s : n->succs()) {
                out[n].insert(s);
            }
        });
    } while(is_unequal_map(tmp_in, in) || is_unequal_map(tmp_out, out));
}

//------------------------------------------------------------------------------

}
