#include "thorin/def.h"

namespace thorin {

class DependencyChecker {
public:
    DependencyChecker(const Param* param)
        : param_(param)
    {}

    bool depends(const Def* def) {
        if (def->is_const()) return false;
        if (!done_.emplace(def).second) return false;

        for (auto op : def->extended_ops()) {
            if (depends(op)) return true;
        }

        return false;
    }

private:
    const Param* param_;
    DefSet done_;
};

bool depends(const Def* def, const Param* param) {
    DependencyChecker checker(param);
    return checker.depends(def);
}

}
