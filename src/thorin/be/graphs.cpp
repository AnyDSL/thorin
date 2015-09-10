#include "thorin/be/graphs.h"

#include "thorin/analyses/cfg.h"
#include "thorin/analyses/dfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

// TODO get rid of this global var
int YCompConfig::INDENTATION = 0;

void YCompCommandLine::add(std::string graph, bool temp, std::string file) {
    graphs.push_back(graph);
    temps.push_back(temp);
    files.push_back(file);
}

void YCompCommandLine::print(World& world) {
    for(unsigned int i = 0; i < graphs.size(); i++) {
        std::string graph = graphs[i];
        const bool temp = temps[i];
        std::ofstream file(files[i]);

        if(graph.compare("domtree") == 0) {
            DomTree::emit_world(world, file);
        } else if(graph.compare("cfg") == 0) {
            if(temp) {
                CFG<true>::emit_world(world, file);
            } else {
                CFG<false>::emit_world(world, file);
            }
        } else if(graph.compare("dfg") == 0) {
            if(temp) {
                DFGBase<true>::emit_world(world, file);
            } else {
                DFGBase<false>::emit_world(world, file);
            }
        } else if(graph.compare("looptree") == 0) {
            if(temp) {
                LoopTree<true>::emit_world(world, file);
            } else {
                LoopTree<false>::emit_world(world, file);
            }
        } else {
            std::cerr << "No outpur for " << graph << " found!" << std::endl;
        }
    }
}

}
