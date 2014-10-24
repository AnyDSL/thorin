#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class Liveness {
public:
    Liveness(const Schedule& schedule)
        : schedule_(schedule)
        , domtree_(*scope().domtree())
    {}

    const Schedule& schedule() const { return schedule_; }
    const Scope& scope() const { return schedule_.scope(); }
    const DomTree& domtree() const { return domtree_; }

private:
    const Schedule& schedule_;
    const DomTree& domtree_;
};

}
