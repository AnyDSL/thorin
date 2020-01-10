#include "thorin/def.h"

namespace thorin {

class DepChecker {
public:
    DepChecker(const Def* on)
        : on_(on)
    {}

    bool depends(const Def* def) {
        if (def->is_const() || !done_.emplace(def).second) return false;

        for (auto op : def->extended_ops()) {
            if (depends(op)) return true;
        }

        return false;
    }

private:
    const Def* on_;
    DefSet done_;
};

bool depends(const Def* def, const Def* on) {
    DepChecker checker(on);
    return checker.depends(def);
}

}
