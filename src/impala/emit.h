#ifndef IMPALA_EMIT_H
#define IMPALA_EMIT_H

namespace thorin {
    class World;
}

namespace impala {

class Scope;

bool emit(thorin::World&, const Scope*);

} // namespace impala

#endif // IMPALA_EMIT_H
