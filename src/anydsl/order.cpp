#include "anydsl/order.h"

#include <stack>

#include "anydsl/def.h"

namespace anydsl {

PostOrder::PostOrder(const Def* def) {
    init(def, 0);
}

PostOrder::~PostOrder() {
    reset();
}

void PostOrder::reset() {
    indices_.clear();
    list_.clear();
}

bool PostOrder::visited(const Def* def) const {
    return indices_.find(def) != indices_.end();
}

void PostOrder::init(const Def* current, int baseIndex) {
    std::stack<const Def*> nodeStack;
    for(;;) {
        if(!visited(current)) {
            // insert invalid index
            indices_[current] = -1;
            const size_t numops = current->size();
            if(numops > 0) {
                nodeStack.push(current);
                for(size_t index = 0; index < numops - 1; ++index) {
                    const Def* currentop = current->op(index);
                    if(!visited(currentop))
                        nodeStack.push(currentop);
                }
                const Def* nextop = current->op(numops - 1);
                if(!visited(nextop))
                    current = current->op(numops - 1);
                else {
                    current = nodeStack.top();
                    nodeStack.pop();
                }
            } else
                continue;
        } else {
            list_.push_back(current);
            indices_[current] = baseIndex++;
            if(nodeStack.empty())
                break;
            current = nodeStack.top();
            nodeStack.pop();
        }
    }
}

}
