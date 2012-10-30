#include "anydsl2/transform/cfg_builder.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class CFGBuilder {
public:

    CFGBuilder(Lambda* entry)
        : scope(entry)
    {}

    void transform();

private:

    Scope scope;
};

void CFGBuilder::transform() {
    for_all (lambda, scope.rpo().slice_back(1)) {
        if (lambda->is_ho()) {
            size_t size = lambda->num_params();
            Array<size_t>  indices(size);
            Array<const Def*> with(size);

            for_all (use, lambda->uses()) {
                if (use.index() != 0)
                    continue;
                if (Lambda* ulambda = use.def()->isa_lambda()) {
                    size_t num = 0;
                    for (size_t i = 0; i != size; ++i) {
                        if (lambda->param(i)->type()->is_ho()) {
                            indices[num] = i;
                            with[num++]  = ulambda->arg(i);
                        }
                    }

                    Lambda* dropped = lambda->drop(indices.slice_front(num), with.slice_front(num), true);
                    ulambda->jump(dropped, ulambda->args().cut(indices.slice_front(num)));
                    scope.reassign_sids();
                }
            }
        }
    }
}

void cfg_transform(Lambda* entry) {
    CFGBuilder builder(entry);
    builder.transform();
}

} // namespace anydsl2
