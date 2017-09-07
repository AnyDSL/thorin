#ifndef THORIN_TRANSFORM_AGGREGATE_REPLICATION_H
#define THORIN_TRANSFORM_AGGREGATE_REPLICATION_H

namespace thorin {

class World;

/**
 * Tries to split @p Slot%s containing aggregates.
 */
void aggregate_replication(World&);

}

#endif
