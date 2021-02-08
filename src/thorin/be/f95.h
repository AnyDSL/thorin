#ifndef THORIN_BE_F95_H
#define THORIN_BE_F95_H

#include <iosfwd>

namespace thorin {

class World;

void emit_f95_int(World&, std::ostream& ostream);

}

#endif
