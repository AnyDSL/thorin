#ifndef THORIN_TRANSFORM_PARTIAL_EVALUATION_H
#define THORIN_TRANSFORM_PARTIAL_EVALUATION_H

namespace thorin {

class World;

bool partial_evaluation(World&, bool lower2cff = false);

}

#endif
