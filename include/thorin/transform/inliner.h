#ifndef THORIN_TRANSFORM_INLINER_H
#define THORIN_TRANSFORM_INLINER_H

namespace thorin {

class World;

/**
 * Forces inlining of all callees within @p scope that are not defined in @p scope.
 * There are at most @p threshold many inlining runs performed.
 * If there still remain functions to be inlined, warnings will be emitted
 */
void force_inline(Scope& scope, int threshold);
void inliner(Thorin&);

}

#endif
