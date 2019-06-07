#ifndef THORIN_PASS_LIVE_H
#define THORIN_PASS_LIVE_H

#include "thorin/analyses/cfg.h"

namespace thorin {

//------------------------------------------------------------------------------

/**
 * Analyzes Thorin IR for liveness. Also considers aggregates.
 * Query whether a @p Def is live or not.
 * Based on Kildall, 1937. A unified approach to global program optimization: http://delivery.acm.org/10.1145/520000/512945/p194-kildall.pdf?
 *    Note: Faster Liveness Analysis exists and will be implemented. At the time of writing, this is a PoC.
 */
class Live : public RuntimeCast<Live> {
public:
    Live(World& world);

private:
    void analyze();
private:
    World& world;
};

//------------------------------------------------------------------------------

}

#endif
