#ifndef THORIN_ANALYSES_BB_SCHEDULE_H
#define THORIN_ANALYSES_BB_SCHEDULE_H

#include <vector>

namespace thorin {

class Lambda;
class Scope;

std::vector<Lambda*> bb_schedule(const Scope& scope);

}

#endif
