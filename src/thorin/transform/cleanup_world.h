#ifndef THORIN_CLEANUP_WORLD_H
#define THORIN_CLEANUP_WORLD_H

namespace thorin {

class World;

void cleanup_world(std::unique_ptr<World>& world);

}

#endif
