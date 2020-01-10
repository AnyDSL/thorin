#ifndef THORIN_ANALYSES_DEPENDS_H
#define THORIN_ANALYSES_DEPENDS_H

namespace thorin {

class Def;

bool depends(const Def* def, const Def* on);

}

#endif
