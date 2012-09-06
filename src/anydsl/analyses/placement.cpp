#include "anydsl/analyses/placement.h"

#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/analyses/domtree.h"

namespace anydsl {

void place(const DomNode* root) {
    for_all (param, root->lambda()->params()) {
        for_all (use, param->uses()){
            if (const PrimOp* op = use.def()->isa<PrimOp>()) {
                //assert(
            }
        }
    }
}

void insert() {
}

} // namespace anydsl
