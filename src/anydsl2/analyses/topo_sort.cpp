#include "anydsl2/analyses/topo_sort.h"

#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class TopoSorter {
};

std::vector<const Def*> topo_sort(const Scope& scope) {
    std::vector<const Def*> result;
    size_t pass = scope.world().new_pass();

    for_all (lambda, scope.rpo()) {
        result.push_back(lambda);

        for_all (param, lambda->params())
            result.push_back(param);
    }

    return result;
}

}
