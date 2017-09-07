#ifndef THORIN_TRANSFORM_SPLIT_SLOTS_H
#define THORIN_TRANSFORM_SPLIT_SLOTS_H

namespace thorin {

class World;

/**
 * Tries to split @p Slot%s that are accessed through constant @p LEA%s.
 */
void split_slots(World&);

}

#endif
