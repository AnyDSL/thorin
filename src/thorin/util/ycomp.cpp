#include "thorin/util/ycomp.h"

#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domfrontier.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

void YCompCommandLine::add(std::string graph, bool temp, std::string file) {
    graphs.push_back(graph);
    temps.push_back(temp);
    files.push_back(file);
}

//------------------------------------------------------------------------------

void YComp::write_ycomp(const char* filename) const {
    std::ofstream file(filename);
    stream_ycomp(file);
}

void YComp::ycomp() const {
    auto filename = std::string(world().name()) + "_" + scope().entry()->unique_name() + "_" + name() + ".vcg";
    write_ycomp(filename.c_str());
}

//------------------------------------------------------------------------------

}
