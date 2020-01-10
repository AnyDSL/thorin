#ifndef THORIN_ANALYSES_DEPENDS_H
#define THORIN_ANALYSES_DEPENDS_H

namespace thorin {

class Def;
class Param;

bool depends(const Def*, const Param*);

}

#endif
