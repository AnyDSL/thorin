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

#define YCOMP(T) \
    if (temp) \
        ycomp<T<true >>(world, file); \
    else \
        ycomp<T<false>>(world, file);

void YCompCommandLine::print(World& world) {
    for(unsigned int i = 0; i < graphs.size(); i++) {
        std::string graph = graphs[i];
        const bool temp = temps[i];
        std::ofstream file(files[i]);

        if (graph.compare("domtree") == 0) {
            YCOMP(DomTreeBase);
        } else if (graph.compare("cfg") == 0) {
            YCOMP(CFG);
        } else if (graph.compare("dfg") == 0) {
            YCOMP(DFGBase);
        } else if (graph.compare("looptree") == 0) {
            YCOMP(LoopTree);
        } else {
            throw std::invalid_argument("no output for graph found");
        }
    }
}

//------------------------------------------------------------------------------

void YComp::write_ycomp(const char* filename) const {
    std::ofstream file(filename);
    stream_ycomp(file);
}

void YComp::ycomp() const {
    auto filename = world().name() + "_" + scope().entry()->unique_name() + "_" + name() + ".vcg";
    write_ycomp(filename.c_str());
}

//------------------------------------------------------------------------------

}
