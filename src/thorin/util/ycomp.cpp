#include "thorin/util/ycomp.h"

#include "thorin/analyses/cfg.h"
#include "thorin/analyses/dfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

// TODO get rid of this global var
int YCompConfig::indentation = 0;

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

        if (graph.compare("domtree") == 0) {
            if (temp) {
                ycomp_world<DomTreeBase<true>>(world, file);
            } else {
                ycomp_world<DomTreeBase<false>>(world, file);
            }
        } else if (graph.compare("cfg") == 0) {
            if (temp) {
                ycomp_world<CFG<true>>(world, file);
            } else {
                ycomp_world<CFG<false>>(world, file);
            }
        } else if (graph.compare("dfg") == 0) {
            if (temp) {
                ycomp_world<DFGBase<true>>(world, file);
            } else {
                ycomp_world<DFGBase<false>>(world, file);
            }
        } else if (graph.compare("looptree") == 0) {
            if (temp) {
                ycomp_world<LoopTree<true>>(world, file);
            } else {
                ycomp_world<LoopTree<false>>(world, file);
            }
        } else {
            std::cerr << "No outpur for " << graph << " found!" << std::endl;
        }
    }
}

void YComp::ycomp(const char* filename) {
    std::ofstream file(filename);
    ycomp(file);
}

}
